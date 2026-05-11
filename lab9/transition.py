from __future__ import annotations

import json
import wave
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import librosa

SRC_DIR = Path(__file__).resolve().parent / "src"
OUT_DIR = Path(__file__).resolve().parent / "results"
TMP_DIR = OUT_DIR / "tmp"

TARGET_SAMPLE_RATE = 22050
N_FFT = 2048
HOP_LENGTH = 256
NOISE_PERCENTILE = 15.0
NOISE_SUBTRACTION_FACTOR = 1.25
NOISE_FLOOR = 0.08
LOCAL_TIME_STEP_SEC = 0.1
LOCAL_FREQ_STEP_HZ = 45.0


@dataclass
class AudioData:
    path: Path
    sample_rate: int
    samples: np.ndarray


def ensure_directories() -> None:
    SRC_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)


def list_audio_files() -> list[Path]:
    files = [path for path in sorted(SRC_DIR.iterdir()) if path.is_file() and path.suffix.lower() in {".wav", ".m4a"}]
    return files


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


def load_audio(path: Path) -> AudioData:
    sample_rate, samples = read_audio_mono(path)
    if sample_rate != TARGET_SAMPLE_RATE:
        samples = resample_signal(samples, sample_rate, TARGET_SAMPLE_RATE)
        sample_rate = TARGET_SAMPLE_RATE
    samples = normalize_signal(samples)
    return AudioData(path=path, sample_rate=sample_rate, samples=samples)


def pad_for_frames(samples: np.ndarray, frame_length: int, hop_length: int) -> np.ndarray:
    if len(samples) < frame_length:
        return np.pad(samples, (0, frame_length - len(samples)))

    remainder = (len(samples) - frame_length) % hop_length
    if remainder == 0:
        return samples
    pad_width = hop_length - remainder
    return np.pad(samples, (0, pad_width))


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


def istft(spectrum: np.ndarray, sample_rate: int, frame_length: int = N_FFT, hop_length: int = HOP_LENGTH, target_length: int | None = None) -> np.ndarray:
    window = np.hanning(frame_length).astype(np.float32)
    n_frames = spectrum.shape[1]
    output_length = (n_frames - 1) * hop_length + frame_length
    signal = np.zeros(output_length, dtype=np.float32)
    normalization = np.zeros(output_length, dtype=np.float32)

    for index in range(n_frames):
        frame = np.fft.irfft(spectrum[:, index], n=frame_length).astype(np.float32)
        start = index * hop_length
        signal[start : start + frame_length] += frame * window
        normalization[start : start + frame_length] += window * window

    nonzero = normalization > 1e-8
    signal[nonzero] /= normalization[nonzero]

    if target_length is not None:
        if len(signal) > target_length:
            signal = signal[:target_length]
        elif len(signal) < target_length:
            signal = np.pad(signal, (0, target_length - len(signal)))

    return signal.astype(np.float32)


def magnitude_to_db(magnitude: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(magnitude, 1e-10))


def plot_spectrogram(frequencies: np.ndarray, times: np.ndarray, magnitude: np.ndarray, title: str, out_path: Path) -> None:
    db = magnitude_to_db(magnitude)
    positive = frequencies > 0.0
    plot_freqs = frequencies[positive]
    plot_db = db[positive]

    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    mesh = ax.pcolormesh(times, plot_freqs, plot_db, shading="auto", cmap="magma")
    ax.set_yscale("log")
    ax.set_ylim(max(1.0, float(plot_freqs[0])), float(plot_freqs[-1]))
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Frequency, Hz")
    ax.set_title(title)
    plt.colorbar(mesh, ax=ax, label="dB")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def estimate_noise_profile(magnitude: np.ndarray) -> np.ndarray:
    profile = np.percentile(magnitude, NOISE_PERCENTILE, axis=1)
    return np.maximum(profile.astype(np.float32), 1e-8)


def reduce_noise(magnitude: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    profile = estimate_noise_profile(magnitude)
    clean = magnitude - NOISE_SUBTRACTION_FACTOR * profile[:, None]
    floor = NOISE_FLOOR * profile[:, None]
    clean = np.maximum(clean, floor)
    return clean.astype(np.float32), profile


def cumulative_sum_2d(matrix: np.ndarray) -> np.ndarray:
    padded = np.pad(matrix, ((1, 0), (1, 0)), mode="constant")
    return padded.cumsum(axis=0).cumsum(axis=1)


def window_sum(csum: np.ndarray, row1: int, row2: int, col1: int, col2: int) -> float:
    return float(csum[row2 + 1, col2 + 1] - csum[row1, col2 + 1] - csum[row2 + 1, col1] + csum[row1, col1])


def find_energy_moments(power: np.ndarray, frequencies: np.ndarray, times: np.ndarray, top_n: int = 8) -> list[dict[str, float]]:
    if power.size == 0 or len(times) == 0 or len(frequencies) < 2:
        return []

    freq_step_hz = max(1.0, float(np.median(np.diff(frequencies[1:])))) if len(frequencies) > 2 else float(frequencies[-1] - frequencies[0])
    time_step_sec = float(np.median(np.diff(times))) if len(times) > 1 else LOCAL_TIME_STEP_SEC

    time_step_frames = max(1, int(round(LOCAL_TIME_STEP_SEC / time_step_sec)))
    freq_step_bins = max(1, int(round(LOCAL_FREQ_STEP_HZ / freq_step_hz)))
    time_radius = max(1, time_step_frames // 2)
    freq_radius = max(1, freq_step_bins // 2)

    csum = cumulative_sum_2d(power)
    candidates: list[dict[str, float]] = []
    for time_index in range(0, power.shape[1], time_step_frames):
        time1 = max(0, time_index - time_radius)
        time2 = min(power.shape[1] - 1, time_index + time_radius)
        for freq_index in range(1, power.shape[0], freq_step_bins):
            freq1 = max(1, freq_index - freq_radius)
            freq2 = min(power.shape[0] - 1, freq_index + freq_radius)
            energy = window_sum(csum, freq1, freq2, time1, time2)
            candidates.append(
                {
                    "time_sec": float(times[time_index]),
                    "frequency_hz": float(frequencies[freq_index]),
                    "energy": energy,
                }
            )

    candidates.sort(key=lambda item: item["energy"], reverse=True)
    selected: list[dict[str, float]] = []
    min_time_gap = LOCAL_TIME_STEP_SEC * 0.75
    min_freq_gap = LOCAL_FREQ_STEP_HZ * 0.75
    for candidate in candidates:
        if all(
            abs(candidate["time_sec"] - chosen["time_sec"]) > min_time_gap
            or abs(candidate["frequency_hz"] - chosen["frequency_hz"]) > min_freq_gap
            for chosen in selected
        ):
            selected.append(candidate)
        if len(selected) >= top_n:
            break

    return selected


def save_energy_map(power: np.ndarray, frequencies: np.ndarray, times: np.ndarray, out_path: Path, title: str, moments: list[dict[str, float]]) -> None:
    db = magnitude_to_db(np.sqrt(power))
    positive = frequencies > 0.0
    plot_freqs = frequencies[positive]
    plot_db = db[positive]

    plt.figure(figsize=(12, 5))
    ax = plt.gca()
    mesh = ax.pcolormesh(times, plot_freqs, plot_db, shading="auto", cmap="viridis")
    ax.set_yscale("log")
    ax.set_ylim(max(1.0, float(plot_freqs[0])), float(plot_freqs[-1]))
    ax.set_xlabel("Time, s")
    ax.set_ylabel("Frequency, Hz")
    ax.set_title(title)
    plt.colorbar(mesh, ax=ax, label="dB")
    if moments:
        xs = [moment["time_sec"] for moment in moments]
        ys = [moment["frequency_hz"] for moment in moments]
        ax.scatter(xs, ys, c="red", s=30, marker="o")
        for index, moment in enumerate(moments, start=1):
            ax.text(moment["time_sec"], moment["frequency_hz"], str(index), color="white", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def write_wav(path: Path, sample_rate: int, samples: np.ndarray) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def process_file(path: Path) -> dict[str, object]:
    audio = load_audio(path)
    frequencies, times, spectrum = stft(audio.samples, audio.sample_rate)
    magnitude = np.abs(spectrum).astype(np.float32)

    before_spec_path = OUT_DIR / f"{path.stem}_spectrogram_before.png"
    plot_spectrogram(frequencies, times, magnitude, f"Spectrogram before noise reduction - {path.name}", before_spec_path)

    clean_magnitude, noise_profile = reduce_noise(magnitude)
    phase = np.exp(1j * np.angle(spectrum))
    clean_spectrum = clean_magnitude * phase
    restored = istft(clean_spectrum, audio.sample_rate, target_length=len(audio.samples))

    restored_audio = normalize_signal(restored)
    restored_path = OUT_DIR / f"{path.stem}_restored.wav"
    write_wav(restored_path, audio.sample_rate, restored_audio)

    clean_frequencies, clean_times, clean_spectrum_preview = stft(restored_audio, audio.sample_rate)
    clean_magnitude_preview = np.abs(clean_spectrum_preview).astype(np.float32)
    after_spec_path = OUT_DIR / f"{path.stem}_spectrogram_after.png"
    plot_spectrogram(clean_frequencies, clean_times, clean_magnitude_preview, f"Spectrogram after noise reduction - {path.name}", after_spec_path)

    power = magnitude**2
    moments = find_energy_moments(power, frequencies, times)
    energy_map_path = OUT_DIR / f"{path.stem}_energy_moments.png"
    save_energy_map(power, frequencies, times, energy_map_path, f"High-energy moments - {path.name}", moments)

    rms_before = float(np.sqrt(np.mean(audio.samples**2))) if len(audio.samples) else 0.0
    rms_after = float(np.sqrt(np.mean(restored_audio**2))) if len(restored_audio) else 0.0
    noise_rms = float(np.sqrt(np.mean((noise_profile / max(np.max(noise_profile), 1e-8)) ** 2)))

    return {
        "file": path.name,
        "sample_rate": audio.sample_rate,
        "duration_sec": float(len(audio.samples) / audio.sample_rate if audio.sample_rate else 0.0),
        "spectrogram_before": before_spec_path.name,
        "spectrogram_after": after_spec_path.name,
        "restored_audio": restored_path.name,
        "rms_before": rms_before,
        "rms_after": rms_after,
        "noise_profile_rms": noise_rms,
        "high_energy_moments": moments,
    }


def main() -> None:
    ensure_directories()
    audio_files = list_audio_files()
    if not audio_files:
        summary = {
            "status": "no_input_files",
            "message": "Put a recording in lab9/src as .wav or .m4a and rerun transition.py.",
        }
        (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(summary["message"])
        return

    summaries = [process_file(path) for path in audio_files]
    (OUT_DIR / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Processed {len(summaries)} file(s)")


if __name__ == "__main__":
    main()