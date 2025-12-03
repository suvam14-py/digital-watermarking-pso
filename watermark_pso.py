# watermark_pso.py
import numpy as np
import pywt
import cv2
from scipy.fftpack import dct, idct
from utils import (img_to_gray_array, psnr, normalized_correlation, xor_encrypt_bits,
                   xor_decrypt_bits, apply_gaussian_noise, apply_jpeg_compression)
from multiprocessing import Pool, cpu_count
import functools

# ---------- DWT & DCT helpers ----------
def dwt2_gray(img):
    coeffs2 = pywt.dwt2(img, 'haar')
    LL, (LH, HL, HH) = coeffs2
    return LL, (LH, HL, HH)

def idwt2_gray(LL, coeffs):
    return pywt.idwt2((LL, coeffs), 'haar')

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# block-wise processing
def block_iter(mat, block_size):
    h, w = mat.shape
    for r in range(0, h, block_size):
        for c in range(0, w, block_size):
            r2 = min(r + block_size, h)
            c2 = min(c + block_size, w)
            yield r, r2, c, c2

# ---------- Embedding & Extraction ----------
def embed_in_channel(channel, watermark, alpha, block_size=8, coef_idx=(3,3)):
    """
    channel: 2D float array (grayscale or Y channel)
    watermark: 2D array same or resized to HH size (binary -1/+1 or 0/1)
    alpha: scalar
    returns embedded_channel (float 0..255)
    """
    # Single-level DWT
    LL, (LH, HL, HH) = dwt2_gray(channel)
    target = HH.copy()
    h, w = target.shape
    # prepare watermark mapped to target size: expected watermark already resized by caller
    wm = watermark.copy().astype(np.float32)
    # map watermark to -1/+1 if binary 0/1
    if wm.max() > 1:
        # assume 0..255 image -> binarize at 127
        wm = (wm > 127).astype(np.float32) * 2 - 1
    else:
        wm = (wm > 0.5).astype(np.float32) * 2 - 1

    out_target = np.zeros_like(target)
    i_coef, j_coef = coef_idx

    for r, r2, c, c2 in block_iter(target, block_size):
        block = target[r:r2, c:c2]
        # pad to full block if needed
        bh, bw = block.shape
        pad_h = 0 if bh == block_size else block_size - bh
        pad_w = 0 if bw == block_size else block_size - bw
        if pad_h or pad_w:
            blockp = np.pad(block, ((0,pad_h),(0,pad_w)), mode='symmetric')
        else:
            blockp = block
        B = dct2(blockp)
        # determine watermark value for this block
        wm_block = wm[r:r2, c:c2]
        wm_val = np.mean(wm_block)
        # modify coefficient (use magnitude factoring to be adaptive)
        orig = B[i_coef,j_coef]
        # avoid zero causing no change
        magnitude = np.abs(orig) if np.abs(orig) > 1e-6 else 1.0
        B[i_coef,j_coef] = orig + alpha * wm_val * magnitude
        block_rec = idct2(B)
        # trim padding
        block_rec = block_rec[:bh, :bw]
        out_target[r:r2, c:c2] = block_rec

    # inverse DWT
    HH_emb = out_target
    channel_emb = idwt2_gray(LL, (LH, HL, HH_emb))
    channel_emb = np.clip(channel_emb, 0, 255)
    return channel_emb

def embed_watermark(cover_img, watermark_img, alpha, mode='grayscale', encrypt=False, key=None):
    """
    cover_img: RGB numpy uint8 (H,W,3) OR grayscale array (H,W)
    watermark_img: grayscale array resized appropriately by caller (2D, values 0..255)
    mode: 'grayscale' or 'color' (color embeds in luminance Y)
    encrypt: bool -> XOR watermark bits before embedding
    key: key for XOR
    returns watermarked_img (same shape/type as cover_img), and mask (if encrypted) for later decrypt
    """
    if mode == 'color' and cover_img.ndim == 3:
        # convert to YCrCb, embed into Y channel
        img_ycc = cv2.cvtColor(cover_img, cv2.COLOR_RGB2YCrCb).astype(np.float32)
        Y = img_ycc[...,0]
        # resize watermark to HH size (we will compute HH size inside embed_in_channel; but we want watermark in HH dims)
        # To get HH dimension: do DWT on Y to know HH shape
        LL, (LH, HL, HH) = dwt2_gray(Y)
        target_shape = HH.shape
        wm_resized = cv2.resize(watermark_img.astype(np.float32), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
        # encryption
        mask = None
        if encrypt:
            bits = (wm_resized > 127).astype(np.uint8)
            enc_bits, mask = xor_encrypt_bits(bits, key)
            # map back to 0/255 for embedding function's binarize logic
            wm_for = (enc_bits * 255).astype(np.float32)
        else:
            wm_for = wm_resized
        Y_emb = embed_in_channel(Y, wm_for, alpha)
        img_ycc[...,0] = Y_emb
        out_rgb = cv2.cvtColor(img_ycc.astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        return out_rgb, mask, wm_resized  # return mask and original resized watermark for NC computation
    else:
        # grayscale embedding
        if cover_img.ndim == 3:
            # convert to grayscale
            cover_gray = cv2.cvtColor(cover_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            cover_gray = cover_img.astype(np.float32)
        LL, (LH, HL, HH) = dwt2_gray(cover_gray)
        target_shape = HH.shape
        wm_resized = cv2.resize(watermark_img.astype(np.float32), (target_shape[1], target_shape[0]), interpolation=cv2.INTER_CUBIC)
        mask = None
        if encrypt:
            bits = (wm_resized > 127).astype(np.uint8)
            enc_bits, mask = xor_encrypt_bits(bits, key)
            wm_for = (enc_bits * 255).astype(np.float32)
        else:
            wm_for = wm_resized
        gray_emb = embed_in_channel(cover_gray, wm_for, alpha)
        # return same type as input: grayscale numpy float32
        return gray_emb.astype(np.uint8), mask, wm_resized

def extract_from_channel(channel_w, channel_orig, wm_shape, block_size=8, coef_idx=(3,3)):
    """
    channel_w: attacked watermarked channel (float)
    channel_orig: original cover channel (float)
    wm_shape: desired watermark shape (H,W) = HH size used during embedding
    returns extracted watermark float 0..255
    """
    _, (_, _, HH_w) = dwt2_gray(channel_w)
    _, (_, _, HH_c) = dwt2_gray(channel_orig)
    h, w = HH_w.shape
    extracted = np.zeros((h,w), dtype=np.float32)
    i_coef, j_coef = coef_idx

    for r, r2, c, c2 in block_iter(HH_w, block_size):
        block_w = HH_w[r:r2, c:c2]
        block_c = HH_c[r:r2, c:c2]
        bh, bw = block_w.shape
        pad_h = 0 if bh == block_size else block_size - bh
        pad_w = 0 if bw == block_size else block_size - bw
        if pad_h or pad_w:
            block_wp = np.pad(block_w, ((0,pad_h),(0,pad_w)), mode='symmetric')
            block_cp = np.pad(block_c, ((0,pad_h),(0,pad_w)), mode='symmetric')
        else:
            block_wp = block_w
            block_cp = block_c
        Bw = dct2(block_wp)
        Bc = dct2(block_cp)
        diff = (Bw[i_coef, j_coef] - Bc[i_coef, j_coef]) / (np.abs(Bc[i_coef, j_coef]) + 1e-8)
        extracted[r:r2, c:c2] = diff

    # resize to wm_shape
    ex = cv2.resize(extracted, (wm_shape[1], wm_shape[0]), interpolation=cv2.INTER_CUBIC)
    # normalize to 0..255
    mn, mx = ex.min(), ex.max()
    if mx - mn < 1e-6:
        out = np.zeros_like(ex)
    else:
        out = (ex - mn) / (mx - mn) * 255.0
    return out.astype(np.uint8)

def extract_watermark(watermarked_img, original_cover_img, wm_resized_shape, mode='grayscale', mask=None):
    """
    returns extracted watermark (0..255), and optionally decrypted binary if mask provided
    """
    if mode == 'color' and watermarked_img.ndim == 3:
        y_w = cv2.cvtColor(watermarked_img, cv2.COLOR_RGB2YCrCb).astype(np.float32)[...,0]
        y_c = cv2.cvtColor(original_cover_img, cv2.COLOR_RGB2YCrCb).astype(np.float32)[...,0]
        extracted = extract_from_channel(y_w, y_c, wm_resized_shape)
    else:
        if watermarked_img.ndim == 3:
            w_gray = cv2.cvtColor(watermarked_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
            c_gray = cv2.cvtColor(original_cover_img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        else:
            w_gray = watermarked_img.astype(np.float32)
            c_gray = original_cover_img.astype(np.float32)
        extracted = extract_from_channel(w_gray, c_gray, wm_resized_shape)
    # if mask provided => decrypt to binary image
    if mask is not None:
        bits_enc = (extracted > 127).astype(np.uint8)
        bits = xor_decrypt_bits(bits_enc, mask)
        return (bits * 255).astype(np.uint8)
    return extracted

# ---------- PSO (optimize scalar alpha) ----------
def fitness_alpha(alpha, cover_img, watermark_img, params):
    """
    alpha: scalar
    params: dict with keys:
      - mode: 'grayscale'/'color'
      - encrypt, key
      - attacks: dict {'gaussian':sigma or None, 'jpeg':quality or None}
      - weights: (w_psnr, w_nc)
    Returns negative fitness (because we'll use minimizer in some contexts) or positive score to maximize
    """
    # embed
    watermarked, mask, wm_resized = embed_watermark(cover_img, watermark_img, alpha,
                                                   mode=params['mode'],
                                                   encrypt=params['encrypt'],
                                                   key=params.get('key', None))
    # simulate attacks if provided
    attacked = watermarked.copy()
    if params['attacks'].get('gaussian') is not None:
        sigma = params['attacks']['gaussian']
        attacked = apply_gaussian_noise(attacked, sigma)
    if params['attacks'].get('jpeg') is not None:
        quality = params['attacks']['jpeg']
        attacked = apply_jpeg_compression(attacked, quality)
    # extract
    extracted = extract_watermark(attacked, cover_img, wm_resized.shape, mode=params['mode'], mask=mask if params['encrypt'] else None)
    # compute metrics
    # For PSNR, compare cover and attacked (or watermarked?) - here use attacked vs original for robustness of perceptual quality after attack
    if cover_img.ndim == 3:
        p = psnr(cv2.cvtColor(cover_img, cv2.COLOR_RGB2GRAY), cv2.cvtColor(attacked, cv2.COLOR_RGB2GRAY))
    else:
        p = psnr(cover_img, attacked)
    # For NC compare original resized (binary) and extracted binary
    orig_bin = (wm_resized > 127).astype(np.float32)
    ex_bin = (extracted > 127).astype(np.float32)
    nc = normalized_correlation(orig_bin, ex_bin)
    w_psnr, w_nc = params['weights']
    # scale PSNR: typical PSNR 0..60 -> map to 0..1 by /60
    score = w_psnr * (p / 60.0) + w_nc * nc
    # return score (higher is better)
    return float(score)

def pso_optimize_alpha_parallel(cover_img, watermark_img, params,
                                n_particles=12, n_iter=30, alpha_bounds=(0.0, 10.0)):
    """
    Parallel PSO evaluating fitness across CPU cores.
    Returns best_alpha, best_score, history_list
    """
    rng = np.random.RandomState(42)
    lb, ub = alpha_bounds
    # initialize particles
    particles = rng.uniform(lb, ub, size=(n_particles,))
    velocities = np.zeros_like(particles)
    pbest = particles.copy()
    # compute pbest scores in parallel
    with Pool(processes=min(cpu_count(), n_particles)) as pool:
        scores = pool.map(functools.partial(_eval_particle, cover_img=cover_img, watermark_img=watermark_img, params=params), particles.tolist())
    pbest_scores = np.array(scores, dtype=np.float32)
    gbest_idx = int(np.argmax(pbest_scores))
    gbest = float(pbest[gbest_idx])
    gbest_score = float(pbest_scores[gbest_idx])
    history = []
    # PSO hyperparams
    w = 0.72
    c1 = 1.49
    c2 = 1.49

    for it in range(n_iter):
        # prepare eval list
        eval_list = []
        # update particles
        for i in range(n_particles):
            r1 = rng.rand()
            r2 = rng.rand()
            velocities[i] = w * velocities[i] + c1 * r1 * (pbest[i] - particles[i]) + c2 * r2 * (gbest - particles[i])
            particles[i] += velocities[i]
            particles[i] = np.clip(particles[i], lb, ub)
            eval_list.append(particles[i])
        # evaluate in parallel
        with Pool(processes=min(cpu_count(), n_particles)) as pool:
            scores = pool.map(functools.partial(_eval_particle, cover_img=cover_img, watermark_img=watermark_img, params=params), eval_list)
        scores = np.array(scores, dtype=np.float32)
        # update pbest and gbest
        for i in range(n_particles):
            if scores[i] > pbest_scores[i]:
                pbest_scores[i] = scores[i]
                pbest[i] = particles[i]
                if scores[i] > gbest_score:
                    gbest_score = float(scores[i])
                    gbest = float(particles[i])
        history.append((it, float(gbest), float(gbest_score)))
    return float(gbest), float(gbest_score), history

def _eval_particle(alpha_val, cover_img, watermark_img, params):
    return fitness_alpha(alpha_val, cover_img, watermark_img, params)
