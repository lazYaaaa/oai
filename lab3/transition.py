from pathlib import Path

import numpy as np
import requests
from PIL import Image

ORIGIN = "https://www.slavcorpora.ru"
SAMPLE_ID = "b008ae91-32cf-4d7d-84e4-996144e4edb7"
IMAGE_COUNT = 5


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


def otsu_threshold(gray: np.ndarray) -> int:
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    prob = hist / total

    omega = np.cumsum(prob)
    mu = np.cumsum(prob * np.arange(256))
    mu_t = mu[-1]

    sigma_b2 = (mu_t * omega - mu) ** 2 / np.maximum(omega * (1.0 - omega), 1e-12)
    return int(np.argmax(sigma_b2))


def majority_filter_3x3(binary: np.ndarray) -> np.ndarray:
    bits = (binary > 0).astype(np.uint8)
    padded = np.pad(bits, 1, mode="edge")
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    k = 3
    sums = integral[k:, k:] - integral[:-k, k:] - integral[k:, :-k] + integral[:-k, :-k]
    return np.where(sums >= 5, 255, 0).astype(np.uint8)


def save_image(arr: np.ndarray, path: Path) -> None:
    Image.fromarray(arr).save(path)


def process_one(idx: int, src_path: Path, out_dir: Path) -> None:
    rgb = np.array(Image.open(src_path).convert("RGB"), dtype=np.uint8)
    gray = rgb_to_gray(rgb)

    t = otsu_threshold(gray)
    mono = np.where(gray > t, 255, 0).astype(np.uint8)
    filtered = majority_filter_3x3(mono)
    diff = np.bitwise_xor(mono, filtered)

    gray_path = out_dir / f"grayscale_{idx:02d}.png"
    mono_path = out_dir / f"mono_{idx:02d}.png"
    filt_path = out_dir / f"filtered_{idx:02d}.png"
    diff_path = out_dir / f"difference_xor_{idx:02d}.png"

    save_image(gray, gray_path)
    save_image(mono, mono_path)
    save_image(filtered, filt_path)
    save_image(diff, diff_path)

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
