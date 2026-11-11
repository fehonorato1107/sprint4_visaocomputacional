import re
from jiwer import cer, wer


def normalize(s: str) -> str:
    """Normalização para comparação:
    maiúsculas, apenas letras e dígitos (remove espaços/pontuação).
    """
    return re.sub(r"[^A-Z0-9]", "", (s or "").upper())


def exact_match(pred: str, gt: str) -> int:
    return int(normalize(pred) == normalize(gt))


def compute_cer(pred: str, gt: str) -> float:
    p, g = normalize(pred), normalize(gt)
    if not g:
        return 0.0 if not p else 1.0
    return cer(g, p)


def compute_wer(pred: str, gt: str) -> float:
    p, g = normalize(pred), normalize(gt)
    if not g:
        return 0.0 if not p else 1.0
    return wer(g, p)


def summarize(rows: list) -> dict:
    """rows: [{'gt':..., 'pred':...}, ...]"""
    n = len(rows)
    if n == 0:
        return {
            "n": 0,
            "accuracy": 0.0,
            "mean_cer": 0.0,
            "mean_wer": 0.0,
        }
    return {
        "n": n,
        "accuracy": sum(exact_match(r["pred"], r["gt"]) for r in rows) / n,
        "mean_cer": sum(compute_cer(r["pred"], r["gt"]) for r in rows) / n,
        "mean_wer": sum(compute_wer(r["pred"], r["gt"]) for r in rows) / n,
    }