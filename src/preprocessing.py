"""
Pipeline de pré-processamento para OCR.
Ordem recomendada: grayscale -> CLAHE -> denoise -> deskew -> upscale -> binarização.
"""
import cv2
import numpy as np


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if len(img.shape) == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray, clip_limit: float = 2.0, tile: int = 8) -> np.ndarray:
    """Equalização adaptativa de histograma (iluminação irregular)."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    """Redução de ruído preservando bordas."""
    return cv2.fastNlMeansDenoising(gray, None, h=10,
                                    templateWindowSize=7,
                                    searchWindowSize=21)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Corrige inclinação do texto usando minAreaRect sobre pixels de texto."""
    # Binariza invertido (texto branco em fundo preto)
    thr = cv2.threshold(gray, 0, 255,
                        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thr > 0))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 1.0 or abs(angle) > 45.0:
        return gray  # ângulos extremos indicam erro; ignora
    (h, w) = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def upscale(img: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Upscaling para textos pequenos/borrados."""
    if scale <= 1.0:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * scale), int(h * scale)),
                      interpolation=cv2.INTER_CUBIC)


def binarize(gray: np.ndarray) -> np.ndarray:
    """Binarização de Otsu."""
    _, bw = cv2.threshold(gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return bw


def preprocess(img: np.ndarray,
               use_clahe: bool = True,
               use_denoise: bool = True,
               use_deskew: bool = True,
               upscale_factor: float = 2.0,
               use_binarize: bool = False) -> np.ndarray:
    """
    Pipeline configurável. Binarização desligada por padrão porque
    o EasyOCR costuma ter melhor desempenho em grayscale do que em
    imagens binárias puras.
    """
    out = to_grayscale(img)
    if use_clahe:
        out = apply_clahe(out)
    if use_denoise:
        out = denoise(out)
    if use_deskew:
        out = deskew(out)
    if upscale_factor and upscale_factor > 1.0:
        out = upscale(out, upscale_factor)
    if use_binarize:
        out = binarize(out)
    return out