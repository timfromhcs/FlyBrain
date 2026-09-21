"""Generate FlyBrain .ico icon (multi-size)."""
import os
from PIL import Image, ImageDraw, ImageFont
import math


def make_flybrain_image(size: int) -> Image.Image:
    img = Image.new("RGBA", (size, size), (10, 14, 26, 255))
    d = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    r = int(size * 0.32)
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(0, 180, 220, 255), outline=(0, 230, 255, 255), width=max(2, size // 32))
    for angle in range(0, 360, 60):
        rad = math.radians(angle)
        ex = cx + int((r + int(size * 0.08)) * math.cos(rad))
        ey = cy + int((r + int(size * 0.08)) * math.sin(rad))
        sx = cx + int(r * 0.6 * math.cos(rad))
        sy = cy + int(r * 0.6 * math.sin(rad))
        d.ellipse([ex - size // 18, ey - size // 18, ex + size // 18, ey + size // 18], fill=(255, 100, 60, 255))
        d.line([(sx, sy), (ex, ey)], fill=(255, 200, 50, 255), width=max(2, size // 40))
    for lobe in [-1, 1]:
        lr = int(r * 0.45)
        ox = int(r * 0.15 * lobe)
        d.ellipse([cx + ox - lr, cy - lr + int(size * 0.04), cx + ox + lr, cy + lr + int(size * 0.04)], fill=(20, 90, 160, 255))
    try:
        font = ImageFont.truetype("arial.ttf", max(8, size // 6))
    except Exception:
        font = ImageFont.load_default()
    bbox = d.textbbox((0, 0), "v10", font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text((cx - tw // 2, cy + r + size // 14 - th // 2), "v10", fill=(255, 255, 255, 255), font=font)
    return img


out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
out_dir = os.path.normpath(out_dir)
os.makedirs(out_dir, exist_ok=True)
img256 = make_flybrain_image(256)
img256.save(os.path.join(out_dir, "flybrain_logo.png"))
img256.save(os.path.join(out_dir, "flybrain_logo.ico"))
for s in [128, 64, 48, 32, 16]:
    make_flybrain_image(s).save(os.path.join(out_dir, "flybrain_logo.ico"), append=True)
print("Icon written: assets/flybrain_logo.ico")