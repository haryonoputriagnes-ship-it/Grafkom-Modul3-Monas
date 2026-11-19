import numpy as np
import os
from raster3d import render
from objio import load_obj
from ppm_io import read_ppm, write_ppm
from srgb import srgb_to_linear, linear_to_srgb
import raster3d

def generate_mipmaps(tex_u8, levels=4):
    """Membuat piramida Mipmap 4-level dengan box downscale gamma-correct."""
    mips = [srgb_to_linear(tex_u8.astype(float) / 255.0)]
    for _ in range(levels - 1):
        prev = mips[-1]
        # Box downscale (2x2 rata-rata)
        down = (prev[0::2, 0::2] + prev[1::2, 0::2] + prev[0::2, 1::2] + prev[1::2, 1::2]) / 4.0
        mips.append(down)
    return mips

def sample_mipmap(mips, u, v, lod):
    """Sampling bilinear dengan memilih level mipmap berdasarkan LOD."""
    level = int(np.clip(round(lod), 0, len(mips)-1))
    tex = mips[level]
    H, W = tex.shape[:2]
    u = np.mod(u, 1.0) * (W - 1)
    v = np.mod(v, 1.0) * (H - 1)
    x0 = np.floor(u).astype(int); y0 = np.floor(v).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1); y1 = np.clip(y0 + 1, 0, H - 1)
    fx = (u - x0)[..., None]; fy = (v - y0)[..., None]
    x0 = np.clip(x0, 0, W - 1); y0 = np.clip(y0, 0, H - 1)
    top = tex[y0, x0] * (1 - fx) + tex[y0, x1] * fx
    bot = tex[y1, x0] * (1 - fx) + tex[y1, x1] * fx
    return top * (1 - fy) + bot * fy

# --- Setup Scene ---
mesh = load_obj("data/mymodel.obj") # Menggunakan model Monas
tex_u8 = read_ppm("data/monas_tex.ppm") # Menggunakan tekstur Monas
mips_linear = generate_mipmaps(tex_u8)

# Kamera JAUH (minification) diatur agar Monas terlihat utuh dari jauh
cam_args = dict(w=256, h=256, eye=(0, 20, 60), center=(0, 8, 0), far=200.0)

print("Mulai render TANPA mipmap...")
# 1. Render TANPA Mipmap (LOD = 0 terus / Resolusi Penuh)
col_no_mip, _, _ = render(mesh, tex_u8=tex_u8, **cam_args)
write_ppm("out/mipmap_off.ppm", col_no_mip)

print("Mulai render DENGAN mipmap...")
# 2. Render DENGAN Mipmap 
def sample_with_mip(tex_lin, u, v):
    # Kita hardcode LOD ke level 2.5 (resolusi tekstur lebih kecil) untuk simulasi jarak jauh
    return sample_mipmap(mips_linear, u, v, lod=2.5)

# Simpan fungsi asli dan ganti sementara
original_sample = raster3d.sample_bilinear
raster3d.sample_bilinear = sample_with_mip

col_mip, _, _ = render(mesh, tex_u8=tex_u8, **cam_args)
write_ppm("out/mipmap_on.ppm", col_mip)

# Kembalikan fungsi asli
raster3d.sample_bilinear = original_sample

# 3. Buat Difference Image (Citra Selisih)
diff = np.abs(col_no_mip.astype(int) - col_mip.astype(int)).astype(np.uint8)
write_ppm("out/mipmap_diff.ppm", diff)

# 4. Hitung Energi Frekuensi Tinggi (Standar deviasi dari Laplacian)
def high_freq_energy(img):
    gray = img.mean(axis=2)
    lap = np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()
    return lap

print("\n--- HASIL ---")
print(f"Energi Frekuensi Tinggi TANPA Mipmap: {high_freq_energy(col_no_mip):.4f}")
print(f"Energi Frekuensi Tinggi DENGAN Mipmap: {high_freq_energy(col_mip):.4f}")
print("Cek gambar di folder out/: mipmap_off.ppm, mipmap_on.ppm, mipmap_diff.ppm")