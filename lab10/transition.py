from __future__ import annotations

import json
import re
import wave
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import librosa

SRC_DIR = Path(__file__).resolve().parent / "src"
OUT_DIR = Path(__file__).resolve().parent / "results"

TARGET_SAMPLE_RATE = 16000
FRAME_LENGTH = 400
HOP_LENGTH = 160
N_FFT = 1024
MIN_SEGMENT_SEC = 0.16
PADDING_SEC = 0.03
GAP_FILL_SEC = 0.025

TOKEN_ALIASES = {
    "0": "0",
    "ноль": "0",
    "1": "1",
    "один": "1",
    "2": "2",
    "два": "2",
    "3": "3",
    "три": "3",
    "4": "4",
    "четыре": "4",
    "5": "5",
    "пять": "5",
    "6": "6",
    "шесть": "6",
    "7": "7",
    "семь": "7",
    "8": "8",
    "восемь": "8",
    "9": "9",
    "девять": "9",
    "плюс": "плюс",
}




@dataclass
class AudioData:
    path: Path
    sample_rate: int
    samples: np.ndarray


@dataclass
class Template:
    label: str
    features: np.ndarray
    path: Path


def ensure_directories() -> None:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def read_audio_mono(path: Path) -> tuple[int, np.ndarray]:
    if path.suffix.lower() == ".wav":
        with wave.open(str(path), "rb") as wav_file:
            sample_rate = wav_file.getframerate()
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)

        if sample_width == 1:
            data = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
            data = (data - 128.0) / 128.0
        elif sample_width == 2:
            data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")

        if channels > 1:
            data = data.reshape(-1, channels).mean(axis=1)
    else:
        data, sample_rate = librosa.load(str(path), sr=None, mono=True)
        data = data.astype(np.float32)

    return sample_rate, data.astype(np.float32)


def resample_signal(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(samples) == 0:
        return samples.astype(np.float32, copy=False)

    duration = len(samples) / float(src_rate)
    new_length = max(1, int(round(duration * dst_rate)))
    old_positions = np.arange(len(samples), dtype=np.float64) / float(src_rate)
    new_positions = np.arange(new_length, dtype=np.float64) / float(dst_rate)
    return np.interp(new_positions, old_positions, samples).astype(np.float32)


def normalize_signal(samples: np.ndarray) -> np.ndarray:
    if samples.size == 0:
        return samples.astype(np.float32)
    peak = float(np.max(np.abs(samples)))
    if peak <= 0.0:
        return samples.astype(np.float32)
    return (samples / peak).astype(np.float32)


def pre_emphasize(samples: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    if samples.size < 2:
        return samples.astype(np.float32, copy=False)
    emphasized = np.empty_like(samples, dtype=np.float32)
    emphasized[0] = samples[0]
    emphasized[1:] = samples[1:] - coefficient * samples[:-1]
    return emphasized


def load_audio(path: Path) -> AudioData:
    sample_rate, samples = read_audio_mono(path)
    if sample_rate != TARGET_SAMPLE_RATE:
        samples = resample_signal(samples, sample_rate, TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE
    samples = normalize_signal(samples)
    return AudioData(path=path, sample_rate=sample_rate, samples=samples)


def normalize_token(text: str) -> str:
    token = text.strip().lower().replace("ё", "е")
    token = re.sub(r"[^a-zа-я0-9+]+", "", token)
    if token == "plus":
        return "плюс"
    return TOKEN_ALIASES.get(token, token)


def is_template_file(path: Path) -> bool:
    token = normalize_token(path.stem)
    return token in set(TOKEN_ALIASES.values())


def list_template_files() -> list[Path]:
    template_dir = SRC_DIR / "templates"
    if template_dir.exists():
        wav_candidates = [path for path in sorted(template_dir.iterdir()) if path.is_file() and path.suffix.lower() == ".wav"]
        m4a_candidates = [path for path in sorted(template_dir.iterdir()) if path.is_file() and path.suffix.lower() == ".m4a"]
        candidates = wav_candidates if wav_candidates else m4a_candidates
    else:
        wav_candidates = [path for path in sorted(SRC_DIR.iterdir()) if path.is_file() and path.suffix.lower() == ".wav"]
        m4a_candidates = [path for path in sorted(SRC_DIR.iterdir()) if path.is_file() and path.suffix.lower() == ".m4a"]
        candidates = wav_candidates if wav_candidates else m4a_candidates
        candidates = [path for path in candidates if normalize_token(path.stem) not in {"phone", "number", "track", "call"}]
    allowed_labels = set(TOKEN_ALIASES.values())
    return [path for path in candidates if is_template_file(path) or normalize_token(path.stem) in allowed_labels]


def list_phone_candidates() -> list[Path]:
    files = [path for path in sorted(SRC_DIR.iterdir()) if path.is_file() and path.suffix.lower() in {".wav", ".m4a"}]
    result = []
    for path in files:
        stem = normalize_token(path.stem)
        if "phone" in stem:
            result.append(path)
    return result


def pad_for_frames(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(samples) < frame_length:
        return np.pad(samples, (0, frame_length - len(samples)))

    remainder = (len(samples) - frame_length) % hop_length
    if remainder == 0:
        return samples
    return np.pad(samples, (0, hop_length - remainder))


def frame_signal(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    padded = pad_for_frames(samples, frame_length, hop_length)
    n_frames = 1 + (len(padded) - frame_length) // hop_length
    frames = np.empty((n_frames, frame_length), dtype=np.float32)
    for index in range(n_frames):
        start = index * hop_length
        frames[index] = padded[start : start + frame_length]
    return frames


def stft(samples: np.ndarray, sample_rate: int, frame_length: int = N_FFT, hop_length: int = HOP_LENGTH) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frames = frame_signal(samples, frame_length, hop_length)
    window = np.hanning(frame_length).astype(np.float32)
    spectrum = np.fft.rfft(frames * window[None, :], n=frame_length, axis=1)
    spectrum = spectrum.T
    frequencies = np.fft.rfftfreq(frame_length, d=1.0 / float(sample_rate))
    times = (np.arange(frames.shape[0], dtype=np.float32) * hop_length + frame_length / 2.0) / float(sample_rate)
    return frequencies.astype(np.float32), times.astype(np.float32), spectrum.astype(np.complex64)


def magnitude_to_db(magnitude: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(magnitude, 1e-10))


def smooth_vector(values: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0 or len(values) == 0:
        return values.astype(np.float32, copy=False)
    kernel = np.ones(2 * radius + 1, dtype=np.float32)
    kernel /= kernel.sum()
    padded = np.pad(values, (radius, radius), mode="edge")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def fill_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    index = 0
    while index < len(out):
        if out[index]:
            index += 1
            continue
        start = index
        while index < len(out) and not out[index]:
            index += 1
        end = index - 1
        if start > 0 and end + 1 < len(out) and out[start - 1] and out[end + 1] and (end - start + 1) <= max_gap:
            out[start : end + 1] = True
    return out


def remove_short_runs(mask: np.ndarray, min_run: int) -> np.ndarray:
    out = mask.copy()
    index = 0
    while index < len(out):
        if not out[index]:
            index += 1
            continue
        start = index
        while index < len(out) and out[index]:
            index += 1
        end = index - 1
        if end - start + 1 < min_run:
            out[start : end + 1] = False
    return out


def detect_segments(samples: np.ndarray, sample_rate: int) -> list[tuple[int, int]]:
    frames = frame_signal(samples, FRAME_LENGTH, HOP_LENGTH)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    rms = smooth_vector(rms, radius=2)

    baseline = float(np.percentile(rms, 20))
    peak = float(np.percentile(rms, 90))
    threshold = max(0.02, baseline + 0.18 * (peak - baseline))

    mask = rms > threshold
    hop_sec = HOP_LENGTH / float(sample_rate)
    mask = fill_gaps(mask, max_gap=max(1, int(round(GAP_FILL_SEC / hop_sec))))
    mask = remove_short_runs(mask, min_run=max(1, int(round(MIN_SEGMENT_SEC / hop_sec))))

    segments: list[tuple[int, int]] = []
    padding_samples = int(round(PADDING_SEC * sample_rate))
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue
        start = index
        while index < len(mask) and mask[index]:
            index += 1
        end = index - 1
        left = max(0, start * HOP_LENGTH - padding_samples)
        right = min(len(samples), end * HOP_LENGTH + FRAME_LENGTH + padding_samples)
        if right > left:
            segments.append((left, right))

    if not segments and len(samples) > 0:
        segments.append((0, len(samples)))

    merged: list[tuple[int, int]] = []
    for start, end in segments:
        if not merged:
            merged.append((start, end))
            continue
        prev_start, prev_end = merged[-1]
        if start <= prev_end + padding_samples:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))

    return merged


def extract_feature_matrix(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    trimmed, _ = librosa.effects.trim(samples, top_db=25, frame_length=N_FFT, hop_length=HOP_LENGTH)
    if len(trimmed) >= max(1, int(round(0.08 * sample_rate))):
        samples = trimmed.astype(np.float32)
    samples = pre_emphasize(samples)

    mfcc = librosa.feature.mfcc(
        y=samples,
        sr=sample_rate,
        n_mfcc=20,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=40,
        fmax=5000.0,
    ).astype(np.float32)
    delta = librosa.feature.delta(mfcc).astype(np.float32)
    delta_delta = librosa.feature.delta(mfcc, order=2).astype(np.float32)
    
    features = np.vstack([mfcc, delta, delta_delta])
    return features.astype(np.float32)


def column_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(1.0, np.sqrt(float(left.size))))


def dtw_distance(left: np.ndarray, right: np.ndarray) -> float:
    if left.size == 0 or right.size == 0:
        return float("inf")

    left_steps = left.shape[1]
    right_steps = right.shape[1]
    if left_steps == 0 or right_steps == 0:
        return float("inf")

    previous = np.full(right_steps + 1, np.inf, dtype=np.float32)
    previous[0] = 0.0

    for left_index in range(1, left_steps + 1):
        current = np.full(right_steps + 1, np.inf, dtype=np.float32)
        for right_index in range(1, right_steps + 1):
            cost = column_distance(left[:, left_index - 1], right[:, right_index - 1])
            current[right_index] = cost + min(previous[right_index], current[right_index - 1], previous[right_index - 1])
        previous = current

    return float(previous[right_steps] / max(1, left_steps + right_steps))


def parse_template_label(path: Path) -> str | None:
    return TOKEN_ALIASES.get(normalize_token(path.stem))


def load_templates() -> list[Template]:
    files = list_template_files()
    templates: list[Template] = []
    for path in files:
        label = parse_template_label(path)
        if label is None:
            continue
        audio = load_audio(path)
        features = extract_feature_matrix(audio.samples, audio.sample_rate)
        templates.append(Template(label=label, features=features, path=audio.path))
    return templates


def choose_phone_file() -> Path | None:
    candidates = list_phone_candidates()
    if candidates:
        wav_candidates = [path for path in candidates if path.suffix.lower() == ".wav"]
        if wav_candidates:
            return wav_candidates[0]
        return candidates[0]
    direct = [path for path in sorted(SRC_DIR.glob("*.wav")) if normalize_token(path.stem) == "phone"]
    return direct[0] if direct else None


def reference_tokens_from_text(text: str) -> list[str]:
    raw = text.strip().lower().replace("ё", "е")
    raw = raw.replace(",", " ").replace(";", " ").replace(".", " ").replace("/", " ")
    if " " in raw:
        parts = [part for part in re.split(r"\s+", raw) if part]
    else:
        parts = list(raw)

    tokens: list[str] = []
    for part in parts:
        token = normalize_token(part)
        if token in {"", "-"}:
            continue
        if token in TOKEN_ALIASES.values():
            tokens.append(token)
    return tokens


def load_reference_tokens() -> list[str] | None:
    reference_path = SRC_DIR / "phone_reference.txt"
    if not reference_path.exists():
        return None
    return reference_tokens_from_text(reference_path.read_text(encoding="utf-8"))


def recognize_segment(segment: np.ndarray, sample_rate: int, templates: dict[str, np.ndarray]) -> tuple[str, float, dict[str, float]]:
    features = extract_feature_matrix(segment, sample_rate)
    
    segment_rms = float(np.sqrt(np.mean(segment**2)))
    
    scores = {}
    for label, template_features in templates.items():
        distance = dtw_distance(features, template_features)
        energy_match = min(1.0, abs(features[-1, 0] - template_features[-1, 0]) + 0.5) if features.shape[0] > 0 and template_features.shape[0] > 0 else 1.0
        adjusted_distance = distance * (0.7 + 0.3 * energy_match)
        scores[label] = 1.0 / (1.0 + adjusted_distance)
    
    if not scores:
        return "", 0.0, {}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_label, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else -1.0
    confidence = max(0.0, min(1.0, 0.5 * (best_score - second_score + 1.0)))
    return best_label, confidence, scores


def plot_waveform_with_segments(samples: np.ndarray, sample_rate: int, segments: list[tuple[int, int]], out_path: Path, title: str) -> None:
    times = np.arange(len(samples), dtype=np.float32) / float(sample_rate)
    plt.figure(figsize=(12, 4))
    plt.plot(times, samples, color="#1f3c88", linewidth=0.9)
    for start, end in segments:
        plt.axvspan(start / float(sample_rate), end / float(sample_rate), color="#f7b32b", alpha=0.18)
    plt.xlabel("Time, s")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_spectrogram(samples: np.ndarray, sample_rate: int, out_path: Path, title: str, segments: list[tuple[int, int]] | None = None) -> None:
    frequencies, times, spectrum = stft(samples, sample_rate)
    magnitude = np.abs(spectrum).astype(np.float32)
    positive = frequencies > 0.0
    plot_freqs = frequencies[positive]
    plot_db = magnitude_to_db(magnitude)[positive]

    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    mesh = ax.pcolormesh(times, plot_freqs, plot_db, shading="auto", cmap="magma")
    ax.set_yscale("log")
    ax.set_ylim(max(1.0, float(plot_freqs[0])), float(plot_freqs[-1]))
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Frequency, Hz")
    ax.set_title(title)
    if segments:
        for start, end in segments:
            ax.axvspan(start / float(sample_rate), end / float(sample_rate), color="white", alpha=0.1)
    plt.colorbar(mesh, ax=ax, label="dB")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def levenshtein(reference: list[str], hypothesis: list[str]) -> int:
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            substitution = previous[j - 1] + (0 if ref_token == hyp_token else 1)
            deletion = previous[j] + 1
            insertion = current[j - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1]


def process_phone_recording(audio: AudioData, templates: dict[str, np.ndarray], reference_tokens: list[str] | None) -> dict[str, object]:
    segments = detect_segments(audio.samples, audio.sample_rate)
    segment_predictions: list[dict[str, object]] = []
    recognized_tokens: list[str] = []
    confidences: list[float] = []

    for index, (start, end) in enumerate(segments, start=1):
        segment = audio.samples[start:end]
        label, confidence, scores = recognize_segment(segment, audio.sample_rate, templates)
        if label:
            recognized_tokens.append(label)
            confidences.append(confidence)
        segment_predictions.append(
            {
                "index": index,
                "start_sec": float(start / audio.sample_rate),
                "end_sec": float(end / audio.sample_rate),
                "label": label,
                "confidence": float(confidence),
                "scores": {k: float(v) for k, v in scores.items()},
            }
        )

    recognized_text = " ".join(recognized_tokens)
    mean_confidence = float(np.mean(confidences)) if confidences else 0.0
    errors = None
    accuracy = None
    if reference_tokens is not None:
        errors = levenshtein(reference_tokens, recognized_tokens)
        accuracy = 1.0 if not reference_tokens else max(0.0, 1.0 - errors / max(1, len(reference_tokens)))

    waveform_path = OUT_DIR / f"{audio.path.stem}_segments.png"
    spectrogram_path = OUT_DIR / f"{audio.path.stem}_spectrogram.png"
    plot_waveform_with_segments(audio.samples, audio.sample_rate, segments, waveform_path, f"Segments - {audio.path.name}")
    plot_spectrogram(audio.samples, audio.sample_rate, spectrogram_path, f"Spectrogram - {audio.path.name}", segments=segments)

    return {
        "phone_file": audio.path.name,
        "segments": segment_predictions,
        "recognized_tokens": recognized_tokens,
        "recognized_text": recognized_text,
        "mean_confidence": mean_confidence,
        "reference_tokens": reference_tokens,
        "errors": errors,
        "accuracy": accuracy,
        "waveform_segments_image": waveform_path.name,
        "spectrogram_image": spectrogram_path.name,
    }


def main() -> None:
    ensure_directories()
    templates_raw = load_templates()
    if not templates_raw:
        summary = {
            "status": "no_templates",
            "message": "Поместите шаблоны цифр в lab10/src/templates/ и отдельную запись слова «плюс».",
        }
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(summary["message"])
        return

    templates = {template.label: template.features for template in templates_raw}
    phone_path = choose_phone_file()
    if phone_path is None:
        summary = {
            "status": "no_phone_recording",
            "message": "Put the phone number recording in lab10/src/ as phone.wav or similar, then rerun transition.py.",
            "templates": sorted(templates.keys()),
        }
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(summary["message"])
        return

    phone_audio = load_audio(phone_path)
    reference_tokens = load_reference_tokens()
    summary = process_phone_recording(phone_audio, templates, reference_tokens)
    summary["available_templates"] = sorted(templates.keys())
    (OUT_DIR / "recognized.txt").write_text(summary["recognized_text"], encoding="utf-8")
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Recognized: {summary['recognized_text']}")


if __name__ == "__main__":
    main()