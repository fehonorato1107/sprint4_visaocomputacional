import csv
import cv2
from pathlib import Path

from preprocessing import preprocess
from ocr_easyocr import read_text


ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "test_set" / "images"
OUT = ROOT / "test_set" / "labels_draft.csv"

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main():
    if not IMAGES.exists():
        print(f"[ERRO] {IMAGES} não existe.")
        return

    files = sorted(p for p in IMAGES.iterdir() if p.suffix.lower() in EXTS)
    if not files:
        print(f"[ERRO] Nenhuma imagem em {IMAGES}.")
        return

    print(f"[i] {len(files)} imagens encontradas. Rodando OCR...\n")

    rows = []
    for p in files:
        img = cv2.imread(str(p))
        if img is None:
            print(f"[!] Falha ao ler {p.name}")
            continue
        try:
            img_prep = preprocess(img, upscale_factor=2.0)
            res = read_text(img_prep)
            text = res["full_text"].replace(",", " ").replace("\n", " ")
        except Exception as e:
            print(f"[!] {p.name}: {e}")
            text = ""
        print(f"  {p.name:20s} -> {text!r}")
        rows.append({"image": p.name, "expected_text": text})

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "expected_text"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n[i] Rascunho salvo em: {OUT}")
    print("[i] Abra o arquivo, confira cada texto contra a imagem e")
    print("    corrija onde o OCR errou. Depois salve como labels.csv")


if __name__ == "__main__":
    main()