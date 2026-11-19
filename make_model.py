"""Module 3 -- Custom Monas Model (Presisi & Menyatu)."""
from __future__ import annotations
import argparse
import os
import numpy as np
from objio import save_obj
from ppm_io import write_ppm
from srgb import linear_to_srgb, to_u8

def create_monas_mesh():
    positions, uvs, faces, face_uvs = [], [], [], []
    
    def add_ring(cy, radius, segments=32):
        start_idx = len(positions)
        for i in range(segments):
            angle = 2.0 * np.pi * i / segments
            x = radius * np.cos(angle)
            z = radius * np.sin(angle)
            positions.append([x, cy, z])
            uvs.append([i / segments, cy])
        return start_idx

    # Profil Monas (y, radius)
    # Dibuat presisi: alas lebar, badan bertingkat, tiang lurus, dan puncak lancip.
    profile = [
        (0.0, 8.0),     # Alas paling bawah (lebar)
        (1.0, 8.0),     # Alas atas
        (1.5, 5.5),     # Tingkat 1 bawah
        (3.5, 5.5),     # Tingkat 1 atas
        (4.0, 3.5),     # Tingkat 2 bawah
        (6.0, 3.5),     # Tingkat 2 atas
        (6.5, 2.0),     # Cawan Bawah
        (7.5, 2.0),     # Cawan Bawah atas
        (8.0, 0.8),     # Tiang Utama bawah
        (13.0, 0.8),    # Tiang Utama atas
        (13.5, 1.5),    # Cawan Atas bawah
        (14.5, 1.5),    # Cawan Atas atas
        (16.0, 0.0),    # Ujung Lidah Api (runcing)
    ]

    seg = 32
    rings = []
    for y, r in profile:
        if r == 0.0:
            idx = len(positions)
            positions.append([0, y, 0])
            uvs.append([0.5, 0.5])
            rings.append(('point', idx))
        else:
            idx = add_ring(y, r, seg)
            rings.append(('ring', idx))

    # Sambungkan antar ring
    for i in range(len(rings) - 1):
        type_bot, idx_bot = rings[i]
        type_top, idx_top = rings[i+1]

        if type_bot == 'ring' and type_top == 'ring':
            for k in range(seg):
                n = (k + 1) % seg
                faces.append((idx_bot + k, idx_bot + n, idx_top + n))
                face_uvs.append((idx_bot + k, idx_bot + n, idx_top + n))
                faces.append((idx_bot + k, idx_top + n, idx_top + k))
                face_uvs.append((idx_bot + k, idx_top + n, idx_top + k))
        elif type_bot == 'ring' and type_top == 'point':
            pt_idx = idx_top
            for k in range(seg):
                n = (k + 1) % seg
                faces.append((idx_bot + k, idx_bot + n, pt_idx))
                face_uvs.append((idx_bot + k, idx_bot + n, pt_idx))

    # Tutup alas bawah
    bot_idx = rings[0][1]
    center_bot = len(positions)
    positions.append([0, 0, 0])
    uvs.append([0.5, 0.5])
    for k in range(seg):
        n = (k + 1) % seg
        faces.append((bot_idx + n, bot_idx + k, center_bot))
        face_uvs.append((bot_idx + n, bot_idx + k, center_bot))

    return np.array(positions), np.array(uvs), faces, face_uvs

def relief_texture(n=256):
    y, x = np.mgrid[0:n, 0:n].astype(np.float64)
    noise = np.sin(x * 0.1) * np.cos(y * 0.15) + np.sin(x * 0.05 + y * 0.05)
    base = 0.7 + 0.2 * noise
    lines = (y % 16) < 2
    base[lines] *= 0.6
    tint = (0.9, 0.9, 0.85)
    lin = np.dstack([base * tint[0], base * tint[1], base * tint[2]])
    return to_u8(linear_to_srgb(np.clip(lin, 0, 1)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    
    positions, uvs, faces, face_uvs = create_monas_mesh()
    obj_path = os.path.join(args.out, "mymodel.obj")
    save_obj(obj_path, positions, uvs, faces, face_uvs)
    save_obj(os.path.join(args.out, "pusaka_model.obj"), positions, uvs, faces, face_uvs)
    
    tex = relief_texture()
    tex_path = os.path.join(args.out, "monas_tex.ppm")
    write_ppm(tex_path, tex)
    write_ppm(os.path.join(args.out, "relief.ppm"), tex)
    
    print(f"SUKSES: {obj_path} ({len(faces)} triangles)")
    print(f"SUKSES: {tex_path}")

if __name__ == "__main__":
    main()