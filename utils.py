# utils.py
import numpy as np
import cv2
from PIL import Image
from math import log10, sqrt

def img_to_gray_array(img):
    """PIL Image -> grayscale numpy float32 0..255"""
    if isinstance(img, np.ndarray):
        arr = img
    else:
        arr = np.array(img.convert("L"))
    return arr.astype(np.float32)

def img_to_rgb_array(img):
    """PIL Image -> RGB numpy uint8"""
    if isinstance(img, np.ndarray):
        arr = img
    else:
        arr = np.array(img.convert("RGB"))
    return arr.astype(np.uint8)

def cv_to_pil_gray(arr):
    return Image.fromarray(arr.astype(np.uint8))

def cv_to_pil_rgb(arr):
    return Image.fromarray(arr.astype(np.uint8))

def psnr(original, modified):
    original = original.astype(np.float32)
    modified = modified.astype(np.float32)
    mse = np.mean((original - modified) ** 2)
    if mse == 0:
        return 100.0
    PIXEL_MAX = 255.0
    return 20 * log10(PIXEL_MAX / sqrt(mse))

def normalized_correlation(w_orig, w_extracted):
    wo = w_orig.flatten().astype(np.float32)
    we = w_extracted.flatten().astype(np.float32)
    if np.linalg.norm(wo) == 0 or np.linalg.norm(we) == 0:
        return 0.0
    return float(np.dot(wo, we) / (np.linalg.norm(wo) * np.linalg.norm(we)))

def xor_encrypt_bits(bits, key):
    """bits: numpy array of 0/1; key: int or string. Returns encrypted bits (0/1)."""
    if isinstance(key, str):
        # convert string key to integer seed
        seed = sum([ord(c) for c in key]) & 0xFFFFFFFF
    else:
        seed = int(key) & 0xFFFFFFFF
    rng = np.random.RandomState(seed)
    mask = rng.randint(0, 2, size=bits.shape).astype(np.uint8)
    return np.bitwise_xor(bits.astype(np.uint8), mask), mask

def xor_decrypt_bits(bits_enc, mask):
    return np.bitwise_xor(bits_enc.astype(np.uint8), mask).astype(np.uint8)

def apply_gaussian_noise(image, sigma):
    """image: uint8 numpy, add gaussian noise with std sigma (0..255 scale)"""
    if image.dtype != np.uint8:
        img = image.astype(np.uint8)
    else:
        img = image
    noise = np.random.normal(0, sigma, img.shape)
    out = img.astype(np.float32) + noise
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def apply_jpeg_compression(image, quality):
    """image: uint8 numpy (BGR or grayscale). quality: 1..100"""
    # use cv2.imencode to simulate JPEG compression
    ext = '.jpg'
    result, encimg = cv2.imencode(ext, image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not result:
        return image
    decimg = cv2.imdecode(encimg, cv2.IMREAD_UNCHANGED)
    if decimg is None:
        return image
    return decimg
