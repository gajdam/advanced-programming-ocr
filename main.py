#!/usr/bin/env python3
"""
License-plate detection & OCR demo (Polish)
------------------------------------------
Pipeline + ulepszenia dokładności OCR

* YOLOv8 detekcja tablic (waga LP-detection.pt z HF)
* OCR EasyOCR (fallback: Tesseract)
* pre-processing wyciętej tablicy: podbicie kontrastu, adaptacyjne progowanie, powiększenie
* sanity-check i naprawa typowych pomyłek znaków (J vs )​, 0↔O, 5↔S, …)
* ignorowanie prefiksu „PL” (flaga UE)
* porównanie z ground-truth w `annotations.xml`
* log „na żywo” z nazwą pliku, tekstem OCR i prawidłową wartością
* **opcjonalny podgląd** tablicy po preprocessingu – flaga `--show`
"""

# TODO: fix image displaying

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

PLATE_REGEX = re.compile(r"^[A-Z]{1,3}[A-Z0-9]{3,5}$")


def sanitize(text: str) -> str:
    """Czyści OCR-owy tekst zgodnie z regułami tablic PL."""
    text = text.upper().strip()
    text = re.sub(r"[^A-Z0-9]", "", text)              # usuń spacje i inne znaki
    text = "".join(CONFUSION_MAP.get(ch, ch) for ch in text)

    if text.startswith("PL") and len(text) > 5:
        text = text[2:]

    return text


def load_ocr() -> easyocr.Reader:
    """Ładuje EasyOCR bez GPU (działa też na CPU-only)."""
    return easyocr.Reader(["pl"], gpu=True)


def preprocess_plate(
    img: np.ndarray
) -> np.ndarray:
    """
    Preprocessing obrazu tablicy do OCR.

    :param img: obraz wejściowy (kolorowy)
    :param for_ocr: 'easyocr' (domyślnie) lub 'tesseract'
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



# global flaga pozwalająca wyłączyć podgląd po ESC/q
SHOW_IMAGES = True


def plate_text(
    reader: easyocr.Reader, img: np.ndarray, show: bool = False
) -> Tuple[str, np.ndarray]:
    """
    Zwraca (zsanityzowany_tekst, obraz_po_preprocessingu).

    Jeśli show=True – pokazuje `proc` w oknie OpenCV.
    Użytkownik może wyłączyć dalsze wyświetlanie klawiszem ESC lub q.
    """
    global SHOW_IMAGES

    proc = preprocess_plate(img)

    # ---------- PODGLĄD DEBUG -----------------------------------------
    if show and SHOW_IMAGES:
        win_name = "plate-debug"
        cv2.imshow(win_name, proc)
        key = cv2.waitKey(0) & 0xFF
        # exit debug mode
        if key in (27, ord("q")):
            cv2.destroyWindow(win_name)
            SHOW_IMAGES = False
    # ------------------------------------------------------------------

    result = reader.readtext(proc, detail=0, paragraph=False)

    return sanitize(result[0] if result else ""), proc


# -------------------- XML -------------------------------------
def load_annotations(xml_path: Path) -> Dict[str, str]:
    """Parsuje plik CVAT-style `annotations.xml` i zwraca mapę file → plate_text."""
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
    limit: int | None = None,
    show: bool = False,
) -> None:
    detector = load_detector()
    reader = load_ocr()

    correct, total = 0, 0
    images = images if limit is None else images[:limit]

    for img_path in images:
        start = time.perf_counter()
        img = cv2.imread(str(img_path))
        if img is None:
            logger.warning("Skipping %s (cannot read)", img_path)
            continue

        img_h, img_w = img.shape[:2]

        # --- YOLO detekcja tablicy ---
        results = detector(img, conf=0.15, iou=0.5)
        boxes = results[0].boxes.xyxy.cpu().numpy() if results else []
        if not len(boxes):
            logger.info(
                "%s | OCR: <brak detekcji> | GT: %s",
                img_path.name,
                annotations.get(img_path.name, ""),
            )
            continue

        # bierzemy najsilniejszą detekcję + margines
        x1, y1, x2, y2 = boxes[0].astype(int)
        x1, y1, x2, y2 = enlarge_box(x1, y1, x2, y2, img_w, img_h)

        plate_crop = img[y1:y2, x1:x2]
        ocr_txt, _ = plate_text(reader, plate_crop, show=show)
        gt_txt = annotations.get(img_path.name, "")

        match = ocr_txt == gt_txt and PLATE_REGEX.match(ocr_txt) is not None
        correct += match
        total += 1

        logger.info(
            "%s | OCR: %-10s | GT: %-10s | %s",
            img_path.name,
            ocr_txt,
            gt_txt,
            "✓" if match else "✗",
        )
        logger.debug("Processed in %.3f s", time.perf_counter() - start)

    accuracy = correct / total * 100 if total else 0.0
    logger.info("\nSummary: %d/%d correct (%.2f%%)", correct, total, accuracy)


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
        "--num", type=int, default=None, help="Limit liczby obrazów (opcjonalnie)"
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

    ann = load_annotations(args.annotations)

    evaluate(img_files, ann, args.num, show=args.show)

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
