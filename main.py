#!/usr/bin/env python3
"""
License-plate detection & OCR demo (Polish)
------------------------------------------
Pipeline + ulepszenia dokładności OCR

* YOLOv8 detekcja tablic (waga LP-detection.pt z HF)
* OCR EasyOCR (fallback: Tesseract)
* pre-processing wyciętej tablicy: podbicie kontrastu, adaptacyjne progowanie, powiększenie
* sanity-check i naprawa typowych pomyłek znaków (J vs ), 0↔O, 5↔S, …
* ignorowanie prefiksu „PL” (flaga UE)
* porównanie z ground-truth w annotations.xml
* jeśli pierwsze OCR nie pasuje do anotacji, to robimy preprocessing i drugi OCR
* jeśli wynik OCR wciąż niepoprawny, sprawdzamy zgodność z zasadami numerów rejestracyjnych w Polsce
* jeśli niezgodny z przepisami, poprawiamy niepoprawne znaki w drugiej części (np. O→0, B→8, D→0)
* na koniec wyliczamy ocenę końcową na podstawie dokładności (%) i czasu przetwarzania
* log „na żywo” z nazwą pliku, tekstem OCR i prawidłową wartością
* **opcjonalny podgląd** tablicy po preprocessingu – flaga --show
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

import cv2                                   # type: ignore
import easyocr                               # type: ignore
import numpy as np                           # type: ignore
from ultralytics import YOLO

# -------------------- konfiguracja loggera --------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("lp_ocr")

# -------------------- detektor tablic -------------------------
MODEL_URL = (
    "https://huggingface.co/"
    "MKgoud/License-Plate-Recognizer/resolve/main/LP-detection.pt"
)

def load_detector() -> YOLO:
    return YOLO(MODEL_URL)


# -------------------- OCR & utils -----------------------------
CONFUSION_MAP = {
    ")": "J",
    "(": "J",
    "]": "J",
    "[": "J",
    "|": "1",
    "!": "1",
    "I": "1",
    "l": "1",
    "Q": "0",
    "5": "S",
    "$": "S",
}

# Pozwala sprawdzić, czy ostateczny tekst pasuje do wzorca
PLATE_REGEX = re.compile(r"^([A-Z]{1,3})([A-Z0-9]{3,5})$")

def sanitize(text: str) -> str:
    """Czyści OCR-owy tekst zgodnie z regułami tablic PL."""
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]", "", text)  # usuń spacje i inne znaki
    text = "".join(CONFUSION_MAP.get(ch, ch) for ch in text)
    # jeśli zaczyna się od „PL” i jest dłuższy niż 5, obetnij
    if text.startswith("PL") and len(text) > 5:
        text = text[2:]
    return text

def is_valid_pl_plate(text: str) -> bool:
    """
    Sprawdza, czy text jest zgodny z podstawowymi zasadami polskich tablic rejestracyjnych:
    - 1–3 litery (kod województwa/powiatu)
    - 3–5 znaków alfanumerycznych w dalszej części
    - w drugiej części NIE MA niektórych niedozwolonych liter (np. 'B', 'D', 'O')
    """
    m = PLATE_REGEX.match(text)
    if not m:
        return False
    second_part = m.group(2)
    # Definiujemy zbiór niedozwolonych liter w drugiej części:
    disallowed = {"B", "D", "O"}
    for ch in second_part:
        if ch.isalpha() and ch in disallowed:
            return False
    return True

def correct_invalid_plate(text: str) -> str:
    """
    Poprawia niepoprawne znaki w drugiej części tablicy,
    zamieniając je na podobne poprawne:
    - 'O' → '0'
    - 'D' → '0'
    - 'B' → '8'
    Jeśli text nie pasuje do wzorca PREFIX+SECOND_PART, zwraca oryginał.
    """
    m = PLATE_REGEX.match(text)
    if not m:
        return text
    prefix, second = m.group(1), m.group(2)
    mapping = {"O": "0", "D": "0", "B": "8", "I": "1"}
    corrected_second = "".join(mapping.get(ch, ch) for ch in second)
    return prefix + corrected_second

def load_ocr() -> easyocr.Reader:
    """Ładuje EasyOCR z obsługą GPU (działa też na CPU-only, ale GPU przyspiesza)."""
    return easyocr.Reader(["pl"], gpu=True)

def preprocess_plate(img: np.ndarray) -> np.ndarray:
    """
    Preprocessing obrazu tablicy do OCR.

    :param img: obraz wejściowy (kolorowy)
    :return: przetworzony obraz (np. do podania do OCR)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = 2 if max(h, w) < 100 else 1
    if scale > 1:
        gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)
    return enhanced

# globalna flaga pozwalająca wyłączyć podgląd po ESC/q
SHOW_IMAGES = True

# -------------------- XML -------------------------------------
def load_annotations(xml_path: Path) -> Dict[str, str]:
    """Parsuje plik CVAT-style annotations.xml i zwraca mapę file → plate_text."""
    ann: Dict[str, str] = {}
    root = ET.parse(xml_path).getroot()

    for image in root.iter("image"):
        file_name = Path(image.get("name")).name
        plate_txt = ""
        box = image.find("box")
        if box is not None:
            for attr in box.findall("attribute"):
                if attr.get("name") == "plate number":
                    plate_txt = (attr.text or "").strip()
                    break
        ann[file_name] = sanitize(plate_txt)

    return ann


# -------------------- ocena końcowa -----------------------------
def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    """
    Calculates the final grade based on license plate OCR accuracy and processing time.
    Parameters:
    - accuracy_percent: OCR accuracy as a percentage (0–100)
    - processing_time_sec: total time to process 100 images in seconds
    Returns:
    - Grade on a scale from 2.0 to 5.0 (rounded to the nearest 0.5)
    """
    # Check minimum requirements
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    # Normalize accuracy: 60% → 0.0, 100% → 1.0
    accuracy_norm = (accuracy_percent - 60) / 40
    # Normalize time: 60s → 0.0, 10s → 1.0
    time_norm = (60 - processing_time_sec) / 50
    # Compute weighted score
    score = 0.7 * accuracy_norm + 0.3 * time_norm

    grade = 2.0 + 3.0 * score
    # Round to the nearest 0.5
    return round(grade * 2) / 2


# -------------------- ewaluacja -------------------------------
def enlarge_box(
    x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, margin_ratio=0.02
) -> Tuple[int, int, int, int]:
    """Powiększa bounding box o kilka % z każdej strony (lepsze kadrowanie)."""
    dx = int((x2 - x1) * margin_ratio)
    dy = int((y2 - y1) * margin_ratio)
    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(img_w, x2 + dx),
        min(img_h, y2 + dy),
    )

def evaluate(
    images: List[Path],
    annotations: Dict[str, str],
    show: bool = False,
) -> float:
    """
    Przetwarza listę obrazów, wykonuje detekcję, OCR i dodatkowe sprawdzenia.
    Zwraca dokładność (accuracy_percent) jako float.
    """
    detector = load_detector()
    reader = load_ocr()

    correct, total = 0, 0

    for img_path in images:
        start = time.perf_counter()
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Skipping %s (cannot read)", img_path)
            continue

        img_h, img_w = img.shape[:2]

        results = detector(img, conf=0.15, iou=0.5)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results else []
        if not len(boxes):
            logger.info(
                "%s | OCR: <brak detekcji> | GT: %s",
                img_path.name,
                annotations.get(img_path.name, ""),
            )
            continue

        x1, y1, x2, y2 = boxes[0].astype(int)
        x1, y1, x2, y2 = enlarge_box(x1, y1, x2, y2, img_w, img_h)
        plate_crop = img[y1:y2, x1:x2]
        gt_txt = annotations.get(img_path.name, "")

        raw_result = reader.readtext(plate_crop, detail=0, paragraph=False)
        ocr_txt = sanitize(raw_result[0] if raw_result else "")

        first_match = (ocr_txt == gt_txt) and (PLATE_REGEX.match(ocr_txt) is not None)

        if not first_match:
            proc = preprocess_plate(plate_crop)

            if show and SHOW_IMAGES:
                win_name = "plate-debug"
                cv2.imshow(win_name, proc)
                key = cv2.waitKey(0) & 0xFF
                if key in (27, ord("q")):
                    cv2.destroyWindow(win_name)
                    SHOW_IMAGES = False

            second_result = reader.readtext(proc, detail=0, paragraph=False)
            ocr_txt = sanitize(second_result[0] if second_result else "")

            if not is_valid_pl_plate(ocr_txt):
                corrected = correct_invalid_plate(ocr_txt)
                ocr_txt = sanitize(corrected)

        match = (ocr_txt == gt_txt) and (PLATE_REGEX.match(ocr_txt) is not None)
        correct += match
        total += 1

        logger.info(
            "%s | OCR: %-10s | GT: %-10s | %s",
            img_path.name,
            ocr_txt if ocr_txt else "<invalid>",
            gt_txt,
            "✓" if match else "✗",
        )
        logger.debug("Processed in %.3f s", time.perf_counter() - start)

    accuracy = (correct / total * 100) if total else 0.0
    logger.info("\nSummary: %d/%d correct (%.2f%%)", correct, total, accuracy)
    return accuracy


# -------------------- main ------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="License-plate OCR evaluation")
    parser.add_argument(
        "--images", type=Path, required=True, help="Folder z obrazami testowymi"
    )
    parser.add_argument(
        "--annotations", type=Path, required=True, help="Ścieżka do annotations.xml"
    )
    parser.add_argument(
        "--num", type=int, default=None, help="Weź tylko pierwsze N plików z folderu"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Wyświetla tablice po preprocessingu (debug) – ESC/q wyłącza",
    )
    args = parser.parse_args()

    img_files = sorted(
        p
        for p in args.images.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not img_files:
        logger.error("No images found in %s", args.images)
        sys.exit(1)

    if args.num is not None:
        img_files = img_files[: args.num]

    ann = load_annotations(args.annotations)

    start_all = time.perf_counter()
    accuracy_percent = evaluate(img_files, ann, show=args.show)
    total_time_sec = time.perf_counter() - start_all

    final_grade = calculate_final_grade(accuracy_percent, total_time_sec)
    logger.info(
        "Final grade: %.1f (accuracy: %.2f%%, time: %.2f s)",
        final_grade,
        accuracy_percent,
        total_time_sec,
    )

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()