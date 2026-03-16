from pathlib import Path

import numpy as np
import requests
from PIL import Image

ORIGIN = "https://www.slavcorpora.ru"
SAMPLE_ID = "b008ae91-32cf-4d7d-84e4-996144e4edb7"
IMAGE_COUNT = 5
THRESHOLD = 70

KERNEL_GX = np.array([
    [17, 61, 17],
    [0, 0, 0],
    [-17, -61, -17],
], dtype=np.float32)

KERNEL_GY = np.array([
    [-17, 0, 17],
    [-61, 0, 61],
    [-17, 0, 17],
], dtype=np.float32)


def list_local_images(src_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    files = [p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files)


def download_sample_images(src_dir: Path, count: int) -> list[Path]:
    sample_data = requests.get(f"{ORIGIN}/api/samples/{SAMPLE_ID}", timeout=30).json()
    image_urls = [f"{ORIGIN}/images/{page['filename']}" for page in sample_data["pages"][:count]]

    saved = []
    for i, url in enumerate(image_urls):
        dst = src_dir / f"original_{i:02d}.png"
        data = requests.get(url, timeout=60)
        data.raise_for_status()
        dst.write_bytes(data.content)
        saved.append(dst)
    return saved


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    gray = np.tensordot(rgb.astype(np.float32), weights, axes=([2], [0]))
    return np.clip(gray, 0, 255).astype(np.uint8)


def convolve_3x3(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    src = gray.astype(np.float32)
    h, w = src.shape
    padded = np.pad(src, 1, mode="edge")

    out = np.zeros((h, w), dtype=np.float32)
    for y in range(3):
        for x in range(3):
            out += kernel[y, x] * padded[y:y + h, x:x + w]
    return out


def normalize_0_255(arr: np.ndarray) -> np.ndarray:
    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


def save_image(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


def process_one(idx: int, src_path: Path, out_dir: Path) -> None:
    rgb = np.array(Image.open(src_path).convert("RGB"), dtype=np.uint8)
    gray = rgb_to_gray(rgb)

    gx = convolve_3x3(gray, KERNEL_GX)
    gy = convolve_3x3(gray, KERNEL_GY)
    g = np.abs(gx) + np.abs(gy)

    gx_n = normalize_0_255(gx)
    gy_n = normalize_0_255(gy)
    g_n = normalize_0_255(g)
    binary = np.where(g_n > THRESHOLD, 255, 0).astype(np.uint8)

    gray_path = out_dir / f"grayscale_{idx:02d}.png"
    gx_path = out_dir / f"gx_{idx:02d}.png"
    gy_path = out_dir / f"gy_{idx:02d}.png"
    g_path = out_dir / f"g_{idx:02d}.png"
    b_path = out_dir / f"binary_{idx:02d}_t{THRESHOLD}.png"

    save_image(gray, gray_path)
    save_image(gx_n, gx_path)
    save_image(gy_n, gy_path)
    save_image(g_n, g_path)
    save_image(binary, b_path)

    return None


def main() -> None:
    base = Path(__file__).resolve().parent
    src_dir = base / "src"
    out_dir = base / "results"
    src_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    local = list_local_images(src_dir)
    if len(local) < IMAGE_COUNT:
        local = download_sample_images(src_dir, IMAGE_COUNT)
    else:
        local = local[:IMAGE_COUNT]

    for i, path in enumerate(local):
        process_one(i, path, out_dir)


if __name__ == "__main__":
    main()
