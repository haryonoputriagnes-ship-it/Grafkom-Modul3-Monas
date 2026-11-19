"""Module 3 -- the from-scratch rasteriser (FIXED Kamera Monas)."""
from __future__ import annotations
import argparse
import os
import numpy as np
from transforms import perspective, look_at, rot_y, translate, scale
from objio import load_obj
from ppm_io import read_ppm, write_ppm
from srgb import srgb_to_linear, linear_to_srgb

def sample_bilinear(tex_lin, u, v):
    H, W = tex_lin.shape[:2]
    u = np.mod(u, 1.0) * (W - 1)
    v = np.mod(v, 1.0) * (H - 1)
    x0 = np.floor(u).astype(int); y0 = np.floor(v).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1); y1 = np.clip(y0 + 1, 0, H - 1)
    fx = (u - x0)[..., None]; fy = (v - y0)[..., None]
    x0 = np.clip(x0, 0, W - 1); y0 = np.clip(y0, 0, H - 1)
    top = tex_lin[y0, x0] * (1 - fx) + tex_lin[y0, x1] * fx
    bot = tex_lin[y1, x0] * (1 - fx) + tex_lin[y1, x1] * fx
    return top * (1 - fy) + bot * fy

def clip_triangle_near(vc, attrs):
    def inside(p):
        return p[2] + p[3] >= 0.0
    poly = [(vc[i], attrs[i]) for i in range(3)]
    out = []
    n = len(poly)
    for i in range(n):
        cur_v, cur_a = poly[i]
        prv_v, prv_a = poly[(i - 1) % n]
        ci, pi = inside(cur_v), inside(prv_v)
        if ci:
            if not pi:
                t = (prv_v[2] + prv_v[3]) / ((prv_v[2] + prv_v[3]) - (cur_v[2] + cur_v[3]))
                out.append((prv_v + t * (cur_v - prv_v), prv_a + t * (cur_a - prv_a)))
            out.append((cur_v, cur_a))
        elif pi:
            t = (prv_v[2] + prv_v[3]) / ((prv_v[2] + prv_v[3]) - (cur_v[2] + cur_v[3]))
            out.append((prv_v + t * (cur_v - prv_v), prv_a + t * (cur_a - prv_a)))
    if len(out) < 3:
        return []
    tris = []
    for k in range(1, len(out) - 1):
        vv = np.stack([out[0][0], out[k][0], out[k + 1][0]])
        aa = np.stack([out[0][1], out[k][1], out[k + 1][1]])
        tris.append((vv, aa))
    return tris

def render(mesh, *, w, h, tex_u8=None, model=None, tri_colors=None,
           eye=(2.6, 2.0, 3.4), center=(0, 0.7, 0), fovy=45.0,
           near=1.0, far=14.0, light_dir=(0.4, 0.9, 0.6), ambient=0.15,
           cull_back=False, depth_bits=None):
    if model is None: model = np.eye(4)
    eye_pos = np.asarray(eye, float)
    P = perspective(fovy, w / h, near, far)
    V = look_at(eye, center)
    world = (model @ np.c_[mesh.positions, np.ones(len(mesh.positions))].T).T[:, :3]
    clip = (P @ V @ model @ np.c_[mesh.positions, np.ones(len(mesh.positions))].T).T
    tris_pos = mesh.faces
    tris_uv = mesh.triangle_uvs()
    L = np.asarray(light_dir, float); L = L / np.linalg.norm(L)
    tex_lin = None
    if tex_u8 is not None:
        tex_lin = srgb_to_linear(tex_u8.astype(np.float64) / 255.0)
    color = np.zeros((h, w, 3), np.float64)
    depth = np.full((h, w), np.inf)
    filled = 0
    for fi in range(len(tris_pos)):
        vidx = tris_pos[fi]
        vc = clip[vidx]
        wp = world[vidx]
        nrm = np.cross(wp[1] - wp[0], wp[2] - wp[0])
        nl = np.linalg.norm(nrm)
        if nl < 1e-12: continue
        nrm = nrm / nl
        if nrm @ (eye_pos - wp[0]) < 0: nrm = -nrm
        shade = ambient + (1 - ambient) * max(0.0, float(nrm @ L))
        uv = tris_uv[fi]
        attrs = np.c_[uv, np.ones(3)]
        for (cvc, cattr) in (clip_triangle_near(vc, attrs) or []):
            wv = cvc[:, 3]
            ndc = cvc[:, :3] / wv[:, None]
            sx = (ndc[:, 0] * 0.5 + 0.5) * w
            sy = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * h
            sz = ndc[:, 2]
            inv_w = 1.0 / wv
            uvw = cattr[:, :2] * inv_w[:, None]
            minx = max(0, int(np.floor(sx.min())))
            maxx = min(w - 1, int(np.ceil(sx.max())))
            miny = max(0, int(np.floor(sy.min())))
            maxy = min(h - 1, int(np.ceil(sy.max())))
            if minx > maxx or miny > maxy: continue
            ax, ay = sx[0], sy[0]
            bx, by = sx[1], sy[1]
            cx, cy = sx[2], sy[2]
            area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
            if area == 0: continue
            if cull_back and area > 0: continue
            ys, xs = np.mgrid[miny:maxy + 1, minx:maxx + 1]
            px = xs + 0.5; py = ys + 0.5
            lA = ((cx - bx) * (py - by) - (cy - by) * (px - bx)) / area
            lB = ((ax - cx) * (py - cy) - (ay - cy) * (px - cx)) / area
            lC = 1.0 - lA - lB
            inside = (lA >= 0) & (lB >= 0) & (lC >= 0)
            if not inside.any(): continue
            l0, l1, l2 = lA, lB, lC
            frag_z = l0 * sz[0] + l1 * sz[1] + l2 * sz[2]
            if depth_bits is not None:
                q = float(2 ** depth_bits)
                frag_z = np.round((frag_z * 0.5 + 0.5) * q) / q * 2.0 - 1.0
            inv_w_i = l0 * inv_w[0] + l1 * inv_w[1] + l2 * inv_w[2]
            reg = depth[miny:maxy + 1, minx:maxx + 1]
            win = inside & (frag_z < reg)
            if not win.any(): continue
            reg[win] = frag_z[win]
            if tex_lin is not None:
                u_i = (l0 * uvw[0, 0] + l1 * uvw[1, 0] + l2 * uvw[2, 0]) / inv_w_i
                v_i = (l0 * uvw[0, 1] + l1 * uvw[1, 1] + l2 * uvw[2, 1]) / inv_w_i
                tex_rgb = sample_bilinear(tex_lin, u_i, v_i)
            elif tri_colors is not None:
                tex_rgb = np.ones((*lA.shape, 3)) * srgb_to_linear(np.asarray(tri_colors[fi], float) / 255.0)
            else:
                tex_rgb = np.ones((*lA.shape, 3)) * 0.7
            lit = np.clip(tex_rgb * shade, 0, 1)
            cregion = color[miny:maxy + 1, minx:maxx + 1]
            cregion[win] = linear_to_srgb(lit)[win]
            filled += int(win.sum())
    return (np.clip(color * 255, 0, 255).astype(np.uint8), depth, filled)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--w", type=int, default=512)
    ap.add_argument("--h", type=int, default=512)
    ap.add_argument("--out", default="out/temple_raster.ppm")
    args = ap.parse_args()
    if not args.model:
        raise SystemExit("give --model")
    mesh = load_obj(args.model)
    tex = read_ppm("data/monas_tex.ppm") if os.path.exists("data/monas_tex.ppm") else None
    model = translate((0, 0, 0)) @ rot_y(35) @ scale((1.4, 1.4, 1.4))
    
    # KAMERA SUDAH DIPERBAIKI: Mundur ke jarak 30, diarahkan ke tengah Monas (y=12)
    col, dep, filled = render(mesh, w=args.w, h=args.h, tex_u8=tex, model=model,
                              eye=(0, 15.0, 30.0), center=(0, 12.0, 0), 
                              near=0.1, far=100.0)
                              
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    write_ppm(args.out, col)
    print(f"SUKSES: {args.out}")

if __name__ == "__main__":
    main()