from __future__ import annotations

import os
from pathlib import Path
import json

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage import filters, feature, exposure
from skimage.color import rgb2gray
import colorsys

CONTRAST_FACTOR = 1.25
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9

SRC_DIR = Path(__file__).resolve().parent / "src"
OUT_DIR = Path(__file__).resolve().parent / "results"
SRC_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)


def ensure_example_image(path: Path) -> None:
    if any(path.glob("*.png")) or any(path.glob("*.jpg")):
        return
    img = Image.new("RGB", (512, 384), "white")
    draw = Image.new("L", img.size)
    arr = np.asarray(draw, dtype=np.uint8)
    rng = np.random.RandomState(0)
    noise = (rng.normal(loc=128, scale=30, size=arr.shape)).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(np.stack([noise, noise, noise], axis=2))
    img.save(path / "example_texture.png")


def rgb_image_to_hsl_arr(img: Image.Image) -> np.ndarray:
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h = np.zeros(rgb.shape[:2], dtype=np.float32)
    l = np.zeros_like(h)
    s = np.zeros_like(h)
    rows, cols = h.shape
    for i in range(rows):
        for j in range(cols):
            r, g, b = rgb[i, j]
            hh, ll, ss = colorsys.rgb_to_hls(r, g, b)
            h[i, j] = hh
            l[i, j] = ll
            s[i, j] = ss
    return np.stack([h, l, s], axis=2)


def hsl_arr_to_rgb_image(hsl: np.ndarray) -> Image.Image:
    rows, cols, _ = hsl.shape
    rgb = np.zeros((rows, cols, 3), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            hh, ll, ss = hsl[i, j]
            r, g, b = colorsys.hls_to_rgb(hh, ll, ss)
            rgb[i, j, 0] = r
            rgb[i, j, 1] = g
            rgb[i, j, 2] = b
    rgb_img = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(rgb_img)


def linear_brightness_transform(L: np.ndarray, factor: float) -> np.ndarray:
    out = (factor * (L - 0.5)) + 0.5
    return np.clip(out, 0.0, 1.0)


def compute_sobel_edge(L: np.ndarray) -> np.ndarray:
    mag = filters.sobel(L)
    return mag


def compute_hog_histogram(edge_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hog_vec, hog_image = feature.hog(
        edge_img,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        visualize=True,
        feature_vector=True,
    )
    gy, gx = np.gradient(edge_img)
    angles = (np.arctan2(gy, gx) + np.pi) * (180.0 / np.pi)
    angles = angles % 180.0
    magnitudes = np.hypot(gx, gy)
    bins = np.linspace(0.0, 180.0, HOG_ORIENTATIONS + 1)
    hist, _ = np.histogram(angles.flatten(), bins=bins, weights=magnitudes.flatten())
    hist_norm = hist.astype(float)
    if hist_norm.sum() > 0:
        hist_norm /= hist_norm.sum()
    centers = 0.5 * (bins[:-1] + bins[1:])
    return centers, hist_norm


def plot_and_save_histogram(x: np.ndarray, y: np.ndarray, title: str, out_path: Path) -> None:
    plt.figure(figsize=(6, 3))
    plt.bar(x, y, width=(x[1] - x[0]) * 0.9)
    plt.title(title)
    plt.xlabel("Orientation (deg)")
    plt.ylabel("Normalized magnitude")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_brightness_histogram(L: np.ndarray, title: str, out_path: Path) -> None:
    vals = (L.flatten() * 255).astype(np.uint8)
    plt.figure(figsize=(6, 3))
    plt.hist(vals, bins=256, range=(0, 255), color="gray")
    plt.title(title)
    plt.xlabel("Brightness")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def process_image(path: Path) -> dict:
    img = Image.open(path).convert("RGB")
    hsl = rgb_image_to_hsl_arr(img)
    H, L, S = hsl[:, :, 0], hsl[:, :, 1], hsl[:, :, 2]

    edge = compute_sobel_edge(L)
    centers, hist_norm = compute_hog_histogram(edge)
    orig_out = OUT_DIR / (path.stem + "_orig.png")
    img.save(orig_out)
    L_img = Image.fromarray((L * 255).astype(np.uint8))
    L_img.save(OUT_DIR / (path.stem + "_L.png"))
    edge_vis = (exposure.rescale_intensity(edge, out_range=(0, 255))).astype(np.uint8)
    Image.fromarray(edge_vis).save(OUT_DIR / (path.stem + "_edge.png"))
    save_brightness_histogram(L, f"Brightness before - {path.name}", OUT_DIR / (path.stem + "_hist_before.png"))
    L2 = linear_brightness_transform(L, CONTRAST_FACTOR)
    hsl2 = np.stack([H, L2, S], axis=2)
    img2 = hsl_arr_to_rgb_image(hsl2)
    img2.save(OUT_DIR / (path.stem + "_contrasted.png"))
    save_brightness_histogram(L2, f"Brightness after - {path.name}", OUT_DIR / (path.stem + "_hist_after.png"))
    edge2 = compute_sobel_edge(L2)
    centers2, hist_norm2 = compute_hog_histogram(edge2)
    edge2_vis = (exposure.rescale_intensity(edge2, out_range=(0, 255))).astype(np.uint8)
    Image.fromarray(edge2_vis).save(OUT_DIR / (path.stem + "_edge_after.png"))
    plot_and_save_histogram(centers, hist_norm, f"HOG histogram (before) - {path.name}", OUT_DIR / (path.stem + "_hog_before.png"))
    plot_and_save_histogram(centers2, hist_norm2, f"HOG histogram (after) - {path.name}", OUT_DIR / (path.stem + "_hog_after.png"))
    np.save(OUT_DIR / (path.stem + "_edge_matrix.npy"), edge)
    np.save(OUT_DIR / (path.stem + "_edge_matrix_after.npy"), edge2)
    summary = {
        "image": path.name,
        "contrast_factor": float(CONTRAST_FACTOR),
        "hog_before": hist_norm.tolist(),
        "hog_after": hist_norm2.tolist(),
        "mean_brightness_before": float(L.mean()),
        "mean_brightness_after": float(L2.mean()),
    }
    return summary


def main() -> None:
    ensure_example_image(SRC_DIR)
    summaries = []
    for path in sorted(SRC_DIR.glob("*.png")) + sorted(SRC_DIR.glob("*.jpg")):
        print("Processing:", path)
        summaries.append(process_image(path))

    (OUT_DIR / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
