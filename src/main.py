"""
Uso:
  python src/main.py --image test_set/images/foo.jpg --approach B_easyocr_preprocessado
"""
import argparse
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preprocessing import preprocess
from ocr_easyocr import read_text as easy_read
from ocr_multimodal import read_text as florence_read
from visualization import draw_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--approach", default="B_easyocr_preprocessado",
                    choices=["A_easyocr_baseline",
                             "B_easyocr_preprocessado",
                             "C_florence2_multimodal"])
    ap.add_argument("--out", default="results/single_out.jpg")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Imagem não encontrada: {args.image}")

    if args.approach == "A_easyocr_baseline":
        res = easy_read(img)
    elif args.approach == "B_easyocr_preprocessado":
        res = easy_read(preprocess(img, upscale_factor=2.0))
    else:
        res = florence_read(img)

    print("Texto extraído:", res["full_text"])
    draw_boxes(img, res["boxes"], args.out)
    print("BBox salva em:", args.out)


if __name__ == "__main__":
    main()