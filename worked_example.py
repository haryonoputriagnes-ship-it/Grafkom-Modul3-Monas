"""Module 3 -- Fully Worked Example for Monas Model (Vertex 0: Puncak Lidah Api).

Telusuri satu titik dari Object -> World -> View -> Clip -> NDC -> Screen.
Menyesuaikan dengan model 'mymodel.obj' dan kamera di 'raster3d.py'.
"""
from __future__ import annotations

import numpy as np

from transforms import (translate, rot_y, scale, look_at, perspective, project, viewport)

np.set_printoptions(precision=3, suppress=True)

# 1. OBJECT SPACE: Titik puncak lidah api Monas (di file mymodel.obj)
p = np.array([0.0, 16.0, 0.0])
print(f"1. Object p      = {np.r_[p, 1.0]}")

# 2. WORLD SPACE: Transformasi Model (Ry 35 deg & Scale 1.4)
M = translate((0, 0, 0)) @ rot_y(35) @ scale((1.4, 1.4, 1.4))
world = (M @ np.r_[p, 1.0])
print(f"2. World         = M p               = {world}")

# 3. VIEW SPACE: Kamera di (0, 15, 30) menghadap (0, 12, 0)
V = look_at(eye=(0, 15.0, 30.0), center=(0, 12.0, 0))
eye = V @ world
print(f"3. View (Camera) = V world           = {eye}   "
      f"(z={eye[2]:.3f}: {'in front' if eye[2] < 0 else 'BEHIND'})")

# 4. CLIP SPACE: Proyeksi Perspektif
P = perspective(45, 1.0, 0.1, 100.0)
clip = P @ eye
print(f"4. Clip          = P eye             = {clip}")

# 5. NDC SPACE: Perspective Divide (Bagi dengan W)
ndc = clip[:3] / clip[3]
print(f"5. NDC           = clip / clip.w     = {ndc}")

# 6. SCREEN SPACE: Viewport 512x512
W, H = 512, 512
scr = viewport(ndc[None, :], W, H)[0]
print(f"6. Screen ({W}x{H})  = [{scr[0]:.0f}, {scr[1]:.0f}]  depth={(ndc[2] + 1) * 0.5:.3f}")