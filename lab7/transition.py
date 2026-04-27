from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ALPHABET = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
PHRASE = "ΣΕ ΑΓΑΠΩ ΠΑΝΤΑ"
BASE_FONT_SIZE = 52
EXPERIMENT_FONT_SIZE = 58
THRESHOLD = 200
CANVAS_SIZE = (1400, 260)

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


@dataclass
class RecognitionResult:
    hypotheses: list[list[tuple[str, float]]]
    best_string: str
    errors: int
    accuracy_percent: float


def resolve_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if path.exists():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def trim_binary_whitespace(binary: np.ndarray) -> np.ndarray:
    ys, xs = np.where(binary > 0)
    if len(xs) == 0 or len(ys) == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return binary[y_min : y_max + 1, x_min : x_max + 1]


def load_binary_image(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    binary = (arr < THRESHOLD).astype(np.uint8)
    return trim_binary_whitespace(binary)


def render_phrase_binary(text: str, font: ImageFont.ImageFont) -> np.ndarray:
    img = Image.new("L", CANVAS_SIZE, color=255)
    draw = ImageDraw.Draw(img)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (CANVAS_SIZE[0] - text_w) // 2 - bbox[0]
    y = (CANVAS_SIZE[1] - text_h) // 2 - bbox[1]
    draw.text((x, y), text, fill=0, font=font)

    arr = np.array(img, dtype=np.uint8)
    binary = (arr < THRESHOLD).astype(np.uint8)
    return trim_binary_whitespace(binary)


def save_binary_bmp(binary: np.ndarray, path: Path) -> None:
    img = np.where(binary > 0, 0, 255).astype(np.uint8)
    Image.fromarray(img, mode="L").convert("1").save(path)


def draw_boxes_image(binary: np.ndarray, boxes: list[Box], out_path: Path) -> None:
    base = np.where(binary > 0, 0, 255).astype(np.uint8)
    img = Image.fromarray(base, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle([box.x1, box.y1, box.x2, box.y2], outline=(255, 0, 0), width=1)
    img.save(out_path)


def horizontal_profile(binary: np.ndarray) -> np.ndarray:
    return binary.sum(axis=1).astype(int)


def vertical_profile(binary: np.ndarray) -> np.ndarray:
    return binary.sum(axis=0).astype(int)


def fill_small_zero_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    n = len(out)
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
        left_fg = start - 1 >= 0 and out[start - 1]
        right_fg = end + 1 < n and out[end + 1]
        if left_fg and right_fg and gap <= max_gap:
            out[start : end + 1] = True
    return out


def remove_small_fg_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    out = mask.copy()
    n = len(out)
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
    out: list[tuple[int, int]] = []
    i = 0
    n = len(mask)
    while i < n:
        if not mask[i]:
            i += 1
            continue
        start = i
        while i < n and mask[i]:
            i += 1
        end = i - 1
        out.append((start, end))
    return out


def segment_text(binary: np.ndarray) -> tuple[list[Box], list[np.ndarray]]:
    hmask = thin_profile(horizontal_profile(binary), threshold=1, max_gap=1, min_run=2)
    yranges = ranges_from_mask(hmask)

    boxes: list[Box] = []
    crops: list[np.ndarray] = []
    idx = 0

    for y1, y2 in yranges:
        line = binary[y1 : y2 + 1, :]
        vmask = thin_profile(vertical_profile(line), threshold=1, max_gap=1, min_run=2)
        xranges = ranges_from_mask(vmask)

        for x1, x2 in xranges:
            window = line[:, x1 : x2 + 1]
            ys, xs = np.where(window > 0)
            if len(xs) == 0 or len(ys) == 0:
                continue
            yy1 = int(ys.min())
            yy2 = int(ys.max())

            box = Box(idx=idx, x1=x1, y1=y1 + yy1, x2=x2, y2=y1 + yy2)
            crop = trim_binary_whitespace(binary[box.y1 : box.y2 + 1, box.x1 : box.x2 + 1])

            boxes.append(box)
            crops.append(crop)
            idx += 1

    order = sorted(range(len(boxes)), key=lambda i: (boxes[i].y1, boxes[i].x1))
    boxes = [boxes[i] for i in order]
    crops = [crops[i] for i in order]

    for i, box in enumerate(boxes):
        box.idx = i

    return boxes, crops


def compute_feature_vector(binary: np.ndarray) -> np.ndarray:
    h, w = binary.shape
    area = float(h * w)

    ys, xs = np.where(binary > 0)
    mass_count = float(len(xs))
    mass_norm = mass_count / area if area > 0 else 0.0

    if mass_count <= 0:
        cx_norm = 0.0
        cy_norm = 0.0
        ix_norm = 0.0
        iy_norm = 0.0
    else:
        cx = float(xs.mean())
        cy = float(ys.mean())
        cx_norm = cx / (w - 1) if w > 1 else 0.0
        cy_norm = cy / (h - 1) if h > 1 else 0.0

        ix = float(np.sum((ys - cy) ** 2))
        iy = float(np.sum((xs - cx) ** 2))
        ix_norm = ix / (mass_count * (h**2)) if h > 0 else 0.0
        iy_norm = iy / (mass_count * (w**2)) if w > 0 else 0.0

    return np.array([mass_norm, cx_norm, cy_norm, ix_norm, iy_norm], dtype=np.float64)


def euclidean_distance(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.linalg.norm(v1 - v2))


def similarity_from_distance(distance: float) -> float:
    # d=0 => sim=1.0, monotonic decreasing for larger distances.
    return float(1.0 / (1.0 + distance))


def load_template_features(lab5_src: Path) -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for symbol in ALPHABET:
        code = f"u{ord(symbol):04X}"
        path = lab5_src / f"{code}.png"
        if not path.exists():
            raise FileNotFoundError(f"Template not found: {path}")
        binary = load_binary_image(path)
        templates[symbol] = compute_feature_vector(binary)
    return templates


def recognize_symbol(
    symbol_binary: np.ndarray,
    template_features: dict[str, np.ndarray],
) -> list[tuple[str, float]]:
    fv = compute_feature_vector(symbol_binary)
    hypotheses: list[tuple[str, float]] = []

    for symbol, tfv in template_features.items():
        d = euclidean_distance(fv, tfv)
        s = similarity_from_distance(d)
        hypotheses.append((symbol, s))

    hypotheses.sort(key=lambda x: x[1], reverse=True)
    return hypotheses


def evaluate_recognition(hypotheses: list[list[tuple[str, float]]], reference: str) -> RecognitionResult:
    best_chars = [h[0][0] for h in hypotheses if h]
    best_string = "".join(best_chars)

    n = min(len(best_string), len(reference))
    errors = sum(1 for i in range(n) if best_string[i] != reference[i]) + abs(len(best_string) - len(reference))
    accuracy_percent = (100.0 * (len(reference) - errors) / len(reference)) if len(reference) > 0 else 0.0

    return RecognitionResult(
        hypotheses=hypotheses,
        best_string=best_string,
        errors=errors,
        accuracy_percent=accuracy_percent,
    )


def save_hypotheses_text(hypotheses: list[list[tuple[str, float]]], out_path: Path) -> None:
    lines: list[str] = []
    for i, hyps in enumerate(hypotheses, start=1):
        fmt = ", ".join([f"('{ch}', {score:.6f})" for ch, score in hyps])
        lines.append(f"{i}: [{fmt}]")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_hypotheses_json(hypotheses: list[list[tuple[str, float]]], out_path: Path) -> None:
    payload = []
    for i, hyps in enumerate(hypotheses, start=1):
        payload.append(
            {
                "symbol_index": i,
                "hypotheses": [{"symbol": ch, "similarity": float(score)} for ch, score in hyps],
            }
        )
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_boxes_csv(boxes: list[Box], out_path: Path) -> None:
    header = "idx;x1;y1;x2;y2;width;height\n"
    rows = [header]
    for box in boxes:
        width = box.x2 - box.x1 + 1
        height = box.y2 - box.y1 + 1
        rows.append(f"{box.idx};{box.x1};{box.y1};{box.x2};{box.y2};{width};{height}\n")
    out_path.write_text("".join(rows), encoding="utf-8")


def save_summary(
    base_result: RecognitionResult,
    exp_result: RecognitionResult,
    reference: str,
    out_path: Path,
) -> None:
    lines = [
        f"Reference (no spaces): {reference}",
        "",
        "=== Base (font size 52) ===",
        f"Recognized: {base_result.best_string}",
        f"Errors: {base_result.errors}",
        f"Accuracy, %: {base_result.accuracy_percent:.2f}",
        "",
        "=== Experiment (font size 58) ===",
        f"Recognized: {exp_result.best_string}",
        f"Errors: {exp_result.errors}",
        f"Accuracy, %: {exp_result.accuracy_percent:.2f}",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def recognize_line(
    binary: np.ndarray,
    template_features: dict[str, np.ndarray],
    reference_no_spaces: str,
) -> tuple[RecognitionResult, list[Box]]:
    boxes, crops = segment_text(binary)
    hypotheses = [recognize_symbol(crop, template_features) for crop in crops]
    result = evaluate_recognition(hypotheses, reference_no_spaces)
    return result, boxes


def main() -> None:
    base = Path(__file__).resolve().parent
    src_dir = base / "src"
    out_dir = base / "results"

    src_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    repo_root = base.parent
    lab5_src = repo_root / "lab5" / "src"
    lab6_phrase = repo_root / "lab6" / "src" / "phrase.bmp"

    template_features = load_template_features(lab5_src)

    reference_no_spaces = PHRASE.replace(" ", "")

    # Base recognition: use segmented line from lab6.
    if lab6_phrase.exists():
        base_binary = load_binary_image(lab6_phrase)
    else:
        base_binary = render_phrase_binary(PHRASE, resolve_font(BASE_FONT_SIZE))
    save_binary_bmp(base_binary, src_dir / "phrase_base.bmp")

    base_result, base_boxes = recognize_line(base_binary, template_features, reference_no_spaces)
    save_boxes_csv(base_boxes, out_dir / "boxes_base.csv")
    draw_boxes_image(base_binary, base_boxes, out_dir / "segmented_boxes_base.png")
    save_hypotheses_text(base_result.hypotheses, out_dir / "hypotheses_base.txt")
    save_hypotheses_json(base_result.hypotheses, out_dir / "hypotheses_base.json")

    # Experiment: regenerate phrase with different font size.
    experiment_binary = render_phrase_binary(PHRASE, resolve_font(EXPERIMENT_FONT_SIZE))
    save_binary_bmp(experiment_binary, src_dir / "phrase_experiment.bmp")

    exp_result, exp_boxes = recognize_line(experiment_binary, template_features, reference_no_spaces)
    save_boxes_csv(exp_boxes, out_dir / "boxes_experiment.csv")
    draw_boxes_image(experiment_binary, exp_boxes, out_dir / "segmented_boxes_experiment.png")
    save_hypotheses_text(exp_result.hypotheses, out_dir / "hypotheses_experiment.txt")
    save_hypotheses_json(exp_result.hypotheses, out_dir / "hypotheses_experiment.json")

    save_summary(base_result, exp_result, reference_no_spaces, out_dir / "summary.txt")


if __name__ == "__main__":
    main()
