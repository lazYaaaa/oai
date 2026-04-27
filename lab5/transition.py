from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from PIL import Image, ImageDraw, ImageFont

ALPHABET = "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
FONT_CANDIDATES = [
	Path("C:/Windows/Fonts/times.ttf"),
	Path("C:/Windows/Fonts/timesbd.ttf"),
	Path("C:/Windows/Fonts/arial.ttf"),
]
FONT_SIZE = 52
CANVAS_SIZE = (180, 180)
THRESHOLD = 200


@dataclass
class ScalarFeatures:
	symbol: str
	codepoint: str
	width: int
	height: int
	mass_q1: int
	mass_q2: int
	mass_q3: int
	mass_q4: int
	specific_q1: float
	specific_q2: float
	specific_q3: float
	specific_q4: float
	cx: float
	cy: float
	cx_norm: float
	cy_norm: float
	ix: float
	iy: float
	ix_norm: float
	iy_norm: float


def resolve_font() -> ImageFont.FreeTypeFont:
	for path in FONT_CANDIDATES:
		if path.exists():
			return ImageFont.truetype(str(path), FONT_SIZE)
	return ImageFont.load_default()


def render_symbol(symbol: str, font: ImageFont.ImageFont) -> np.ndarray:
	img = Image.new("L", CANVAS_SIZE, color=255)
	draw = ImageDraw.Draw(img)

	bbox = draw.textbbox((0, 0), symbol, font=font)
	text_w = bbox[2] - bbox[0]
	text_h = bbox[3] - bbox[1]
	x = (CANVAS_SIZE[0] - text_w) // 2 - bbox[0]
	y = (CANVAS_SIZE[1] - text_h) // 2 - bbox[1]
	draw.text((x, y), symbol, fill=0, font=font)

	arr = np.array(img, dtype=np.uint8)
	black = arr < THRESHOLD

	ys, xs = np.where(black)
	if len(xs) == 0 or len(ys) == 0:
		return arr

	x_min, x_max = int(xs.min()), int(xs.max())
	y_min, y_max = int(ys.min()), int(ys.max())
	cropped = arr[y_min:y_max + 1, x_min:x_max + 1]
	return cropped


def to_binary_mass(arr: np.ndarray) -> np.ndarray:
	# 1 means black pixel mass, 0 means background.
	return (arr < THRESHOLD).astype(np.uint8)


def quarter_slices(h: int, w: int) -> tuple[slice, slice, slice, slice]:
	hy = h // 2
	wx = w // 2
	q1 = (slice(0, hy), slice(0, wx))
	q2 = (slice(0, hy), slice(wx, w))
	q3 = (slice(hy, h), slice(0, wx))
	q4 = (slice(hy, h), slice(wx, w))
	return q1, q2, q3, q4


def quarter_area(q: tuple[slice, slice]) -> int:
	ys, xs = q
	return max(0, ys.stop - ys.start) * max(0, xs.stop - xs.start)


def compute_scalar_features(symbol: str, mass_img: np.ndarray) -> ScalarFeatures:
	h, w = mass_img.shape
	q1, q2, q3, q4 = quarter_slices(h, w)

	m1 = int(mass_img[q1].sum())
	m2 = int(mass_img[q2].sum())
	m3 = int(mass_img[q3].sum())
	m4 = int(mass_img[q4].sum())

	a1 = quarter_area(q1)
	a2 = quarter_area(q2)
	a3 = quarter_area(q3)
	a4 = quarter_area(q4)

	s1 = float(m1 / a1) if a1 > 0 else 0.0
	s2 = float(m2 / a2) if a2 > 0 else 0.0
	s3 = float(m3 / a3) if a3 > 0 else 0.0
	s4 = float(m4 / a4) if a4 > 0 else 0.0

	ys, xs = np.where(mass_img > 0)
	total_mass = float(len(xs))

	if total_mass <= 0:
		cx = 0.0
		cy = 0.0
		ix = 0.0
		iy = 0.0
	else:
		cx = float(xs.mean())
		cy = float(ys.mean())

		# Central second-order moments as axial moments of inertia.
		ix = float(np.sum((ys - cy) ** 2))
		iy = float(np.sum((xs - cx) ** 2))

	cx_norm = float(cx / (w - 1)) if w > 1 else 0.0
	cy_norm = float(cy / (h - 1)) if h > 1 else 0.0

	ix_norm = float(ix / (total_mass * (h ** 2))) if total_mass > 0 else 0.0
	iy_norm = float(iy / (total_mass * (w ** 2))) if total_mass > 0 else 0.0

	return ScalarFeatures(
		symbol=symbol,
		codepoint=f"u{ord(symbol):04X}",
		width=w,
		height=h,
		mass_q1=m1,
		mass_q2=m2,
		mass_q3=m3,
		mass_q4=m4,
		specific_q1=s1,
		specific_q2=s2,
		specific_q3=s3,
		specific_q4=s4,
		cx=cx,
		cy=cy,
		cx_norm=cx_norm,
		cy_norm=cy_norm,
		ix=ix,
		iy=iy,
		ix_norm=ix_norm,
		iy_norm=iy_norm,
	)


def save_profile_chart(values: np.ndarray, out_path: Path, title: str, x_label: str) -> None:
	indices = np.arange(len(values), dtype=int)

	fig, ax = plt.subplots(figsize=(10, 4), dpi=140)
	ax.bar(indices, values, color="#2A6F9E", edgecolor="black", linewidth=0.4)
	ax.set_title(title)
	ax.set_xlabel(x_label)
	ax.set_ylabel("Black pixels")
	ax.grid(axis="y", alpha=0.25)

	ax.xaxis.set_major_locator(MaxNLocator(integer=True))
	ax.yaxis.set_major_locator(MaxNLocator(integer=True))
	ax.set_xlim(-0.5, len(values) - 0.5)

	if len(values) > 40:
		step = max(1, len(values) // 20)
		ax.set_xticks(indices[::step])

	fig.tight_layout()
	fig.savefig(out_path)
	plt.close(fig)


def write_csv(features: list[ScalarFeatures], csv_path: Path) -> None:
	fields = [
		"symbol",
		"codepoint",
		"width",
		"height",
		"mass_q1",
		"mass_q2",
		"mass_q3",
		"mass_q4",
		"specific_q1",
		"specific_q2",
		"specific_q3",
		"specific_q4",
		"cx",
		"cy",
		"cx_norm",
		"cy_norm",
		"ix",
		"iy",
		"ix_norm",
		"iy_norm",
	]

	with csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
		writer.writeheader()
		for item in features:
			row = {
				"symbol": item.symbol,
				"codepoint": item.codepoint,
				"width": item.width,
				"height": item.height,
				"mass_q1": item.mass_q1,
				"mass_q2": item.mass_q2,
				"mass_q3": item.mass_q3,
				"mass_q4": item.mass_q4,
				"specific_q1": f"{item.specific_q1:.6f}",
				"specific_q2": f"{item.specific_q2:.6f}",
				"specific_q3": f"{item.specific_q3:.6f}",
				"specific_q4": f"{item.specific_q4:.6f}",
				"cx": f"{item.cx:.6f}",
				"cy": f"{item.cy:.6f}",
				"cx_norm": f"{item.cx_norm:.6f}",
				"cy_norm": f"{item.cy_norm:.6f}",
				"ix": f"{item.ix:.6f}",
				"iy": f"{item.iy:.6f}",
				"ix_norm": f"{item.ix_norm:.8f}",
				"iy_norm": f"{item.iy_norm:.8f}",
			}
			writer.writerow(row)


def main() -> None:
	base = Path(__file__).resolve().parent
	src_dir = base / "src"
	out_dir = base / "results"
	profiles_dir = out_dir / "profiles"

	src_dir.mkdir(parents=True, exist_ok=True)
	out_dir.mkdir(parents=True, exist_ok=True)
	profiles_dir.mkdir(parents=True, exist_ok=True)

	font = resolve_font()
	all_features: list[ScalarFeatures] = []

	for symbol in ALPHABET:
		code = f"u{ord(symbol):04X}"
		glyph_img = render_symbol(symbol, font)
		glyph_path = src_dir / f"{code}.png"
		Image.fromarray(glyph_img).save(glyph_path)

		mass_img = to_binary_mass(glyph_img)
		features = compute_scalar_features(symbol, mass_img)
		all_features.append(features)

		profile_x = mass_img.sum(axis=0).astype(int)
		profile_y = mass_img.sum(axis=1).astype(int)

		save_profile_chart(
			profile_x,
			profiles_dir / f"{code}_x.png",
			f"X-profile for {code} ({symbol})",
			"Column index",
		)
		save_profile_chart(
			profile_y,
			profiles_dir / f"{code}_y.png",
			f"Y-profile for {code} ({symbol})",
			"Row index",
		)

	write_csv(all_features, out_dir / "features.csv")


if __name__ == "__main__":
	main()
