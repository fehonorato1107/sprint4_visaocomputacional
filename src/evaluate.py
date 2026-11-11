"""
Executa as 3 abordagens no conjunto de teste e salva:
  - results/predictions.csv    (por imagem, por abordagem)
  - results/summary.csv        (acurácia, CER, WER)
  - results/bbox_{approach}/   (imagens com BBox)
"""
import csv
import io
import cv2
from pathlib import Path
from tqdm import tqdm

from preprocessing import preprocess
from ocr_easyocr import read_text as easy_read
from visualization import draw_boxes
from metrics import summarize

# Florence-2 é opcional — se torchaudio/transformers falharem,
# as abordagens A e B continuam rodando normalmente.
try:
    from ocr_multimodal import read_text as florence_read
    FLORENCE_OK = True
except Exception as _e:
    print(f"[!] Florence-2 indisponível ({type(_e).__name__}: {_e})")
    print("[!] Abordagem C será pulada. A e B continuam.")
    florence_read = None
    FLORENCE_OK = False


ROOT = Path(__file__).resolve().parent.parent
IMAGES = ROOT / "test_set" / "images"
LABELS = ROOT / "test_set" / "labels.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

ABORDAGENS = {
    "A_easyocr_baseline":      "easy_raw",
    "B_easyocr_preprocessado": "easy_prep",
    "C_florence2_multimodal":  "florence",
}


# ---------------------------------------------------------------------------
# Leitura tolerante do labels.csv
# ---------------------------------------------------------------------------
def _build_labels():
    """Lê labels.csv de forma tolerante:
    - Detecta separador (',' ou ';')
    - Remove BOM do UTF-8 (Excel)
    - Aceita variações de nome da coluna de texto esperado
    - Ignora linhas vazias
    """
    if not LABELS.exists():
        print(f"[ERRO] {LABELS} não encontrado.")
        return {}

    with open(LABELS, "r", encoding="utf-8-sig", newline="") as f:
        content = f.read()

    if not content.strip():
        print(f"[ERRO] {LABELS} está vazio.")
        return {}

    # Detecta separador pela primeira linha
    first_line = content.splitlines()[0]
    try:
        dialect = csv.Sniffer().sniff(first_line, delimiters=",;")
        delim = dialect.delimiter
    except csv.Error:
        delim = ","

    reader = csv.DictReader(io.StringIO(content), delimiter=delim)

    # Normaliza nomes das colunas (trim + lowercase)
    cols_norm = {
        (c or "").strip().lower(): c for c in (reader.fieldnames or [])
    }

    candidates_text = [
        "expected_text", "expected", "texto", "texto_esperado",
        "gabarito", "text_expected", "ground_truth",
    ]
    text_col = next(
        (cols_norm[c] for c in candidates_text if c in cols_norm), None
    )
    image_col = cols_norm.get("image")

    if not image_col or not text_col:
        print(f"[ERRO] labels.csv precisa ter colunas 'image' e "
              f"'expected_text'.")
        print(f"       Encontrado: {list(reader.fieldnames or [])}")
        return {}

    rows = {}
    for r in reader:
        img = (r.get(image_col) or "").strip()
        txt = (r.get(text_col) or "").strip()
        if img:
            rows[img] = txt

    print(f"[i] labels.csv carregado ({len(rows)} linhas, sep='{delim}')")
    return rows


# ---------------------------------------------------------------------------
# Wrappers de cada abordagem
# ---------------------------------------------------------------------------
def _run_easy(img, preprocess_flag):
    if preprocess_flag:
        img = preprocess(img, upscale_factor=2.0)
    return easy_read(img)


def _run_florence(img, _flag=None):
    return florence_read(img)


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------
def main():
    labels = _build_labels()

    if len(labels) == 0:
        print("[ERRO] Nenhuma linha válida em labels.csv. Abortando.")
        return

    if len(labels) < 30:
        print(f"[!] Aviso: apenas {len(labels)} imagens no labels.csv "
              f"(esperado 30).")

    if not FLORENCE_OK:
        print("[i] Abordagem C (Florence-2) desativada neste run.")

    all_results = {}  # approach -> [{'approach','image','gt','pred'}, ...]

    for approach, kind in ABORDAGENS.items():
        if kind == "florence" and not FLORENCE_OK:
            continue

        out_dir = RESULTS / f"bbox_{approach}"
        out_dir.mkdir(exist_ok=True)
        rows = []

        for img_name, gt in tqdm(labels.items(), desc=approach):
            img_path = IMAGES / img_name
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[!] Não foi possível ler {img_path}")
                continue

            try:
                if kind == "easy_raw":
                    res = _run_easy(img, preprocess_flag=False)
                elif kind == "easy_prep":
                    res = _run_easy(img, preprocess_flag=True)
                else:
                    res = _run_florence(img)
            except Exception as e:
                print(f"[!] Erro em {img_name} ({approach}): {e}")
                res = {"full_text": "", "boxes": []}

            draw_boxes(img, res["boxes"], out_dir / img_name)

            rows.append({
                "approach": approach,
                "image": img_name,
                "gt": gt,
                "pred": res["full_text"],
            })

        all_results[approach] = rows

    if not all_results:
        print("[ERRO] Nenhuma abordagem executada. Abortando.")
        return

    # CSV por imagem
    with open(RESULTS / "predictions.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["approach", "image", "gt", "pred"]
        )
        w.writeheader()
        for rows in all_results.values():
            w.writerows(rows)

    # Summary
    print("\n========== RESUMO ==========")
    summary_rows = []
    for approach, rows in all_results.items():
        s = summarize(rows)
        s["approach"] = approach
        summary_rows.append(s)
        print(f"{approach:30s} | acc={s['accuracy'] * 100:5.1f}% | "
              f"CER={s['mean_cer']:.3f} | WER={s['mean_wer']:.3f}")

    with open(RESULTS / "summary.csv", "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["approach", "n", "accuracy", "mean_cer", "mean_wer"],
        )
        w.writeheader()
        w.writerows(summary_rows)

    print(f"\nResultados em: {RESULTS}")


if __name__ == "__main__":
    main()