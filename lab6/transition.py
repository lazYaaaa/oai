from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageDraw, ImageFont

ALPHABET = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
PHRASE = "ΣΕ ΑΓΑΠΩ ΠΑΝΤΑ"
FONT_SIZE = 52
CANVAS_SIZE = (1200, 220)
THRESHOLD = 200

FONT_CANDIDATES = [
    Path("C:/Windows/Fonts/times.ttf"),
    Path("C:/Windows/Fonts/timesbd.ttf"),
    Path("C:/Windows/Fonts/arial.ttf"),
]


@dataclass
class Box:
    idx: int
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1 + 1

    @property
    def height(self) -> int:
        return self.y2 - self.y1 + 1


def resolve_font() -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), FONT_SIZE)
    return ImageFont.load_default()


def render_text_to_binary(text: str, font: ImageFont.ImageFont, canvas_size: tuple[int, int]) -> np.ndarray:
    img = Image.new("L", canvas_size, color=255)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (canvas_size[0] - text_w) // 2 - bbox[0]
    y = (canvas_size[1] - text_h) // 2 - bbox[1]
    draw.text((x, y), text, fill=0, font=font)

    arr = np.array(img, dtype=np.uint8)
    black = arr < THRESHOLD

    ys, xs = np.where(black)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    cropped = arr[y_min : y_max + 1, x_min : x_max + 1]

    
    return (cropped < THRESHOLD).astype(np.uint8)


def save_mono_bmp(binary: np.ndarray, out_path: Path) -> None:
    img = np.where(binary > 0, 0, 255).astype(np.uint8)
    pil = Image.fromarray(img, mode="L").convert("1")
    pil.save(out_path)


def horizontal_profile(binary: np.ndarray) -> np.ndarray:
    return binary.sum(axis=1).astype(int)


def vertical_profile(binary: np.ndarray) -> np.ndarray:
    return binary.sum(axis=0).astype(int)


def fill_small_zero_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        start = i
        while i < n and not out[i]:
            i += 1
        end = i - 1
        gap = end - start + 1
        left_is_fg = start - 1 >= 0 and out[start - 1]
        right_is_fg = end + 1 < n and out[end + 1]
        if left_is_fg and right_is_fg and gap <= max_gap:
            out[start : end + 1] = True
    return out


def remove_small_fg_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    out = mask.copy()
    n = len(mask)
    i = 0
    while i < n:
        if not out[i]:
            i += 1
            continue
        start = i
        while i < n and out[i]:
            i += 1
        end = i - 1
        run = end - start + 1
        if run < min_run:
            out[start : end + 1] = False
    return out


def thin_profile(profile: np.ndarray, threshold: int, max_gap: int, min_run: int) -> np.ndarray:
    mask = profile > threshold
    mask = fill_small_zero_gaps(mask, max_gap=max_gap)
    mask = remove_small_fg_runs(mask, min_run=min_run)
    return mask


def ranges_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    n = len(mask)
    i = 0
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        end = i - 1
        ranges.append((start, end))
    return ranges


def segment_line_chars(line_img: np.ndarray, y_offset: int, idx_offset: int = 0) -> list[Box]:
    vprof = vertical_profile(line_img)

    
    vmask = thin_profile(vprof, threshold=1, max_gap=1, min_run=2)
    xranges = ranges_from_mask(vmask)

    boxes: list[Box] = []
    idx = idx_offset

    for x1, x2 in xranges:
        window = line_img[:, x1 : x2 + 1]
        ys, xs = np.where(window > 0)
        if len(xs) == 0 or len(ys) == 0:
            continue
        yy1 = int(ys.min())
        yy2 = int(ys.max())

        box = Box(
            idx=idx,
            x1=x1,
            y1=y_offset + yy1,
            x2=x2,
            y2=y_offset + yy2,
        )
        boxes.append(box)
        idx += 1

    return boxes


def segment_text(binary: np.ndarray) -> list[Box]:
    hprof = horizontal_profile(binary)

    
    hmask = thin_profile(hprof, threshold=1, max_gap=1, min_run=2)
    yranges = ranges_from_mask(hmask)

    boxes: list[Box] = []
    idx = 0

    for y1, y2 in yranges:
        line_img = binary[y1 : y2 + 1, :]
        line_boxes = segment_line_chars(line_img, y_offset=y1, idx_offset=idx)
        boxes.extend(line_boxes)
        idx += len(line_boxes)

    
    boxes.sort(key=lambda b: (b.y1, b.x1))
    for i, box in enumerate(boxes):
        box.idx = i

    return boxes


def draw_boxes(binary: np.ndarray, boxes: list[Box], out_path: Path) -> None:
    base = np.where(binary > 0, 0, 255).astype(np.uint8)
    img = Image.fromarray(base, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)

    for box in boxes:
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=(255, 0, 0), width=1)

    img.save(out_path)


def save_symbol_crops(binary: np.ndarray, boxes: list[Box], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for box in boxes:
        crop = binary[box.y1 : box.y2 + 1, box.x1 : box.x2 + 1]
        img = np.where(crop > 0, 0, 255).astype(np.uint8)
        Image.fromarray(img, mode="L").save(out_dir / f"char_{box.idx:02d}.png")


def save_profile_plot(values: np.ndarray, out_path: Path, title: str, x_label: str) -> None:
    x = np.arange(len(values), dtype=int)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=140)
    ax.bar(x, values, color="#21618C", edgecolor="black", linewidth=0.4)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Black pixels")
    ax.grid(axis="y", alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlim(-0.5, len(values) - 0.5)

    if len(values) > 40:
        step = max(1, len(values) // 20)
        ax.set_xticks(x[::step])

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_boxes_csv(boxes: list[Box], out_path: Path) -> None:
    fields = ["idx", "x1", "y1", "x2", "y2", "width", "height"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        for box in boxes:
            writer.writerow(
                {
                    "idx": box.idx,
                    "x1": box.x1,
                    "y1": box.y1,
                    "x2": box.x2,
                    "y2": box.y2,
                    "width": box.width,
                    "height": box.height,
                }
            )


def save_boxes_json(boxes: list[Box], out_path: Path) -> None:
    serializable = [
        {
            **asdict(box),
            "width": box.width,
            "height": box.height,
        }
        for box in boxes
    ]
    out_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")


def generate_alphabet_profiles(font: ImageFont.ImageFont, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for symbol in ALPHABET:
        glyph = render_text_to_binary(symbol, font, canvas_size=(180, 180))
        code = f"u{ord(symbol):04X}"

        xprof = vertical_profile(glyph)
        yprof = horizontal_profile(glyph)

        save_profile_plot(
            xprof,
            out_dir / f"{code}_x.png",
            f"X-profile for {code} ({symbol})",
            "Column index",
        )
        save_profile_plot(
            yprof,
            out_dir / f"{code}_y.png",
            f"Y-profile for {code} ({symbol})",
            "Row index",
        )


def main() -> None:
    base = Path(__file__).resolve().parent
    src_dir = base / "src"
    out_dir = base / "results"

    src_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    font = resolve_font()

    
    binary = render_text_to_binary(PHRASE, font, CANVAS_SIZE)
    phrase_bmp = src_dir / "phrase.bmp"
    save_mono_bmp(binary, phrase_bmp)

    
    hprof = horizontal_profile(binary)
    vprof = vertical_profile(binary)
    save_profile_plot(hprof, out_dir / "phrase_horizontal_profile.png", "Horizontal profile", "Row index")
    save_profile_plot(vprof, out_dir / "phrase_vertical_profile.png", "Vertical profile", "Column index")

    
    boxes = segment_text(binary)
    draw_boxes(binary, boxes, out_dir / "segmented_boxes.png")
    save_symbol_crops(binary, boxes, out_dir / "segmented_chars")
    save_boxes_csv(boxes, out_dir / "boxes.csv")
    save_boxes_json(boxes, out_dir / "boxes.json")

    
    generate_alphabet_profiles(font, out_dir / "alphabet_profiles")


if __name__ == "__main__":
    main()
