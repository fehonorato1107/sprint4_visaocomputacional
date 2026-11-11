"""Wrapper EasyOCR com parâmetros otimizados para placas/textos curtos."""
import easyocr
import numpy as np

_READER = None


def _get_reader():
    global _READER
    if _READER is None:
        # gpu=True se houver CUDA; fallback automático p/ CPU
        _READER = easyocr.Reader(['en', 'pt'], gpu=True)
    return _READER


def read_text(img: np.ndarray,
              mag_ratio: float = 1.5,
              canvas_size: int = 2560,
              low_text: float = 0.2,
              link_threshold: float = 0.8,
              adjust_contrast: float = 0.7,
              contrast_ths: float = 0.3) -> dict:
    """
    Retorna dict com:
      - full_text: string concatenada (ordenada top-left -> bottom-right)
      - boxes: lista de (bbox, texto, confiança)
    """
    reader = _get_reader()
    results = reader.readtext(
        img,
        detail=1,
        paragraph=False,
        mag_ratio=mag_ratio,
        canvas_size=canvas_size,
        low_text=low_text,
        link_threshold=link_threshold,
        adjust_contrast=adjust_contrast,
        contrast_ths=contrast_ths,
    )

    # Ordena por linha (y) e depois por coluna (x)
    def _key(r):
        box = r[0]
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        return (round(min(ys) / 20), min(xs))

    results.sort(key=_key)

    boxes = [(r[0], r[1], float(r[2])) for r in results]
    full_text = " ".join(r[1] for r in results).strip()
    return {"full_text": full_text, "boxes": boxes}