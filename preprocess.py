"""
preprocess.py
=============
Skrypt preprocessingowy dla magisterki — ujednolicenie trzech baz dźwięków serca.

Źródła danych:
  - PhysioNet Challenge 2016 (sety a–e, bez f)
  - PhysioNet / CirCor Challenge 2022
  - Bentley Dataset B (set_b)

Wynik:
  dataset/
    normal/     <- fizjologia
    murmur/     <- szmer sercowy (patologia)
    extrastole/ <- dodatkowe skurcze (patologia, tylko Bentley)

Każdy plik to wycinek 3 s, próbkowanie 4000 Hz, mono, .wav
"""

import os
import csv
import shutil
import numpy as np
from pathlib import Path

# ── Sprawdź zależności ────────────────────────────────────────────────────────
try:
    import librosa
    import soundfile as sf
except ImportError:
    print("Brak wymaganych bibliotek. Zainstaluj:")
    print("  pip install librosa soundfile")
    raise

# ── Konfiguracja ──────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "dataset"

TARGET_SR       = 4000    # Hz — natywne SR źródeł (2kHz upsampleowane do 4kHz)
SEGMENT_SEC     = 3.0     # długość wycinków w sekundach
SEGMENT_SAMPLES = int(TARGET_SR * SEGMENT_SEC)
MIN_DURATION    = 3.0     # odrzucamy wszystko krótsze niż pełny segment

# Ścieżki do źródeł
P2016_DIR  = BASE_DIR / "physionet.org/files/challenge-2016/1.0.0"
P2022_DIR  = BASE_DIR / "physionet.org/files/circor-heart-sound/1.0.3"
BENTLEY_DIR = BASE_DIR / "set_b"

# ── Funkcje pomocnicze ────────────────────────────────────────────────────────

def load_and_resample(wav_path: Path, target_sr: int = TARGET_SR):
    """Wczytaj plik .wav i resample do target_sr. Zwraca (audio, sr) lub None."""
    try:
        audio, sr = librosa.load(str(wav_path), sr=target_sr, mono=True)
        return audio
    except Exception as e:
        print(f"  [BŁĄD] Nie można wczytać {wav_path.name}: {e}")
        return None


def segment_audio(audio: np.ndarray, seg_samples: int, min_samples: int):
    """
    Podziel audio na wycinki o długości seg_samples.
    Tylko pełne segmenty — reszta odcinana, zero paddingu.
    """
    segments = []
    total = len(audio)
    start = 0
    while start + seg_samples <= total:
        segments.append(audio[start:start + seg_samples])
        start += seg_samples
    return segments


def save_segments(segments, label: str, source_name: str, counter: dict):
    """Zapisz wycinki do OUTPUT_DIR/label/ z unikalną numeracją."""
    out_dir = OUTPUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for seg in segments:
        idx = counter[label]
        counter[label] += 1
        fname = out_dir / f"{label}_{source_name}_{idx:05d}.wav"
        sf.write(str(fname), seg, TARGET_SR, subtype="PCM_16")
        saved += 1
    return saved


def process_file(wav_path: Path, label: str, source: str, counter: dict):
    """Pełny pipeline dla jednego pliku."""
    audio = load_and_resample(wav_path)
    if audio is None:
        return 0
    duration = len(audio) / TARGET_SR
    if duration < MIN_DURATION:
        return 0
    segments = segment_audio(audio, SEGMENT_SAMPLES, SEGMENT_SAMPLES)
    return save_segments(segments, label, source, counter)


# ── Źródło 1: PhysioNet 2016 ──────────────────────────────────────────────────

def load_physionet_2016(counter: dict):
    print("\n[1/3] PhysioNet Challenge 2016 (sety a–e, bez f)")
    sets = ["training-a", "training-b", "training-c", "training-d", "training-e"]
    total = 0

    for set_name in sets:
        set_dir = P2016_DIR / set_name
        ref_csv = set_dir / "REFERENCE.csv"

        if not ref_csv.exists():
            print(f"  Brak REFERENCE.csv w {set_name}, pomijam.")
            continue

        # Wczytaj etykiety: 1 = normal, -1 = abnormal (murmur/inne)
        labels = {}
        with open(ref_csv) as f:
            for row in csv.reader(f):
                if len(row) >= 2:
                    labels[row[0].strip()] = int(row[1].strip())

        n_normal = n_abnormal = n_skip = 0
        for record_id, lbl in labels.items():
            wav_path = set_dir / f"{record_id}.wav"
            if not wav_path.exists():
                n_skip += 1
                continue

            label = "normal" if lbl == 1 else "murmur"  # 2016: abnormal → murmur
            saved = process_file(wav_path, label, f"p2016_{set_name[:1]}", counter)
            total += saved
            if lbl == 1:
                n_normal += saved
            else:
                n_abnormal += saved

        print(f"  {set_name}: normal={n_normal} wycinków, abnormal={n_abnormal} wycinków, brak_pliku={n_skip}")

    print(f"  → Łącznie z 2016: {total} wycinków")
    return total


# ── Źródło 2: PhysioNet 2022 (CirCor) ────────────────────────────────────────

def load_physionet_2022(counter: dict):
    print("\n[2/3] PhysioNet / CirCor Challenge 2022")
    meta_csv = P2022_DIR / "training_data.csv"
    wav_dir  = P2022_DIR / "training_data"
    total = 0

    if not meta_csv.exists():
        print("  Brak training_data.csv, pomijam.")
        return 0

    # Buduj słownik: patient_id → murmur (Present/Absent/Unknown)
    patient_murmur = {}
    with open(meta_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid     = row["Patient ID"].strip()
            murmur  = row["Murmur"].strip()   # Present / Absent / Unknown
            patient_murmur[pid] = murmur

    n_normal = n_murmur = n_skip = 0
    for wav_path in sorted(wav_dir.glob("*.wav")):
        # Nazwa pliku: <patient_id>_<position>.wav, np. 13918_AV.wav
        pid = wav_path.stem.split("_")[0]
        murmur = patient_murmur.get(pid)

        if murmur is None or murmur == "Unknown":
            n_skip += 1
            continue

        label = "normal" if murmur == "Absent" else "murmur"
        saved = process_file(wav_path, label, "p2022", counter)
        total += saved
        if label == "normal":
            n_normal += saved
        else:
            n_murmur += saved

    print(f"  normal={n_normal}, murmur={n_murmur}, pominięto={n_skip}")
    print(f"  → Łącznie z 2022: {total} wycinków")
    return total


# ── Źródło 3: Bentley Dataset B ──────────────────────────────────────────────

def load_bentley(counter: dict):
    print("\n[3/3] Bentley Dataset B (set_b)")
    # Etykieta jest prefiksem nazwy pliku: normal__, murmur__, extrastole__
    # Wersje "noisy" (np. normal_noisy...) pomijamy — niższa jakość
    VALID_LABELS = {"normal", "murmur", "extrastole"}
    total = 0
    stats = {lbl: 0 for lbl in VALID_LABELS}
    n_skip = 0

    for wav_path in sorted(BENTLEY_DIR.glob("*.wav")):
        name = wav_path.stem.lower()

        # Wyciągnij etykietę z prefiksu
        label = None
        for lbl in VALID_LABELS:
            if name.startswith(lbl + "__"):
                label = lbl
                break

        if label is None:
            n_skip += 1  # unlabelled lub noisy
            continue

        saved = process_file(wav_path, label, "bentley", counter)
        total += saved
        stats[label] += saved

    print(f"  normal={stats['normal']}, murmur={stats['murmur']}, "
          f"extrastole={stats['extrastole']}, pominięto(noisy/unlabelled)={n_skip}")
    print(f"  → Łącznie z Bentley: {total} wycinków")
    return total


# ── Raport końcowy ────────────────────────────────────────────────────────────

def print_summary(counter: dict):
    print("\n" + "="*50)
    print("PODSUMOWANIE — dataset/")
    total = sum(counter.values())
    for label, count in sorted(counter.items()):
        bar = "█" * (count // 10)
        print(f"  {label:<12} {count:>5} wycinków  {bar}")
    print(f"  {'RAZEM':<12} {total:>5} wycinków")
    print("="*50)
    print(f"\nPliki zapisane w: {OUTPUT_DIR.resolve()}")
    print("Parametry:")
    print(f"  Próbkowanie : {TARGET_SR} Hz")
    print(f"  Długość     : {SEGMENT_SEC} s")
    print(f"  Próbki/wycinek: {SEGMENT_SAMPLES}")


# ── Główna funkcja ────────────────────────────────────────────────────────────

def main():
    print("="*50)
    print("PREPROCESSING DŹWIĘKÓW SERCA")
    print(f"Output: {OUTPUT_DIR}")
    print("="*50)

    # Wyczyść poprzedni output jeśli istnieje
    if OUTPUT_DIR.exists():
        print(f"\nUsuwam poprzedni dataset ({OUTPUT_DIR})...")
        shutil.rmtree(OUTPUT_DIR)

    counter = {"normal": 0, "murmur": 0, "extrastole": 0}

    load_physionet_2016(counter)
    load_physionet_2022(counter)
    load_bentley(counter)

    print_summary(counter)


if __name__ == "__main__":
    main()
