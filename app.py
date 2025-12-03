# app.py
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from io import BytesIO
import matplotlib.pyplot as plt

from utils import (
    img_to_gray_array,
    img_to_rgb_array,
    cv_to_pil_gray,
    cv_to_pil_rgb,
    apply_gaussian_noise,
    apply_jpeg_compression,
    normalized_correlation,
    psnr as utils_psnr,
)
from watermark_pso import (
    pso_optimize_alpha_parallel,
    embed_watermark,
    extract_watermark,
)

st.set_page_config(page_title="Advanced DWT-DCT Watermarking + PSO", layout="wide")
st.title("Advanced DWT–DCT Watermarking with PSO Optimization")

st.markdown(
    """
A robust DWT-DCT watermarking solution using parallel PSO for global optimization.
Features attack simulation, XOR encryption, and color/grayscale support.
"""
)

# --- Uploads
col_up1, col_up2 = st.columns(2)
with col_up1:
    cover_file = st.file_uploader("Upload cover (RGB) image", type=["png", "jpg", "jpeg"])
with col_up2:
    watermark_file = st.file_uploader(
        "Upload watermark image (grayscale) (optional)", type=["png", "jpg", "jpeg"]
    )

text_watermark = st.text_input("Or watermark text (used if no watermark image provided):", value="Demo WM")

# Processing options
mode = st.radio("Embedding mode:", ("grayscale", "color (luminance Y)"))
resize_opt = st.checkbox("Resize cover to 512×512 (recommended for speed)", value=True)

# Encryption
encrypt = st.checkbox("Enable XOR encryption of watermark (simple)")
key = None
if encrypt:
    key = st.text_input("Encryption key (string or integer):", value="secret")

# Attacks toggles (these are simulated during fitness)
st.subheader("Simulated attacks to include in PSO fitness (robustness)")
gauss_enable = st.checkbox("Include Gaussian noise in fitness", value=True)
gauss_sigma = st.slider("Gaussian sigma (std dev)", 0.0, 50.0, 5.0)
jpeg_enable = st.checkbox("Include JPEG compression in fitness", value=True)
jpeg_quality = st.slider("JPEG quality (1=worst ... 100=best)", 1, 100, 70)

# PSO params
st.subheader("PSO parameters")
n_particles = st.number_input("Particles", min_value=4, max_value=64, value=12, step=1)
n_iter = st.number_input("Iterations", min_value=1, max_value=100, value=20, step=1)
alpha_min = st.number_input("Alpha min", value=0.0, step=0.1)
alpha_max = st.number_input("Alpha max", value=10.0, step=0.1)

# Buttons
run_and_embed = st.button("Run PSO and embed watermark")

# Validate cover
if cover_file is None:
    st.info("Upload a cover image to proceed.")
    st.stop()

# Handle Pillow resizing compatibility
if hasattr(Image, "Resampling"):
    resample_filter = Image.Resampling.LANCZOS  # Pillow 10+
else:
    resample_filter = Image.ANTIALIAS  # Older versions

# Load and optionally resize cover
cover_pil = Image.open(cover_file).convert("RGB")
if resize_opt:
    cover_pil = cover_pil.resize((512, 512), resample_filter)
cover_rgb = img_to_rgb_array(cover_pil)

# Prepare watermark (image or text)
if watermark_file:
    wm_pil = Image.open(watermark_file).convert("L")
else:
    # generate text watermark image sized about 1/4th of cover
    w = max(64, cover_pil.size[0] // 4)
    h = max(64, cover_pil.size[1] // 4)
    img = Image.new("L", (w, h), color=0)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 36)
    except Exception:
        font = ImageFont.load_default()

    # Use textbbox when available (Pillow >= 10), otherwise fallback to textsize
    if hasattr(draw, "textbbox"):
        bbox = draw.textbbox((0, 0), text_watermark, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    else:
        tw, th = draw.textsize(text_watermark, font=font)

    draw.text(((w - tw) // 2, (h - th) // 2), text_watermark, fill=255, font=font)
    wm_pil = img

# Show uploaded inputs
st.image(cover_pil, caption="Cover image", use_column_width=False, width=300)
st.image(wm_pil, caption="Watermark (input)", use_column_width=False, width=150)

# Run PSO & embed
if run_and_embed:
    st.info("PSO running — using parallel workers. This may take 10s..minutes depending on settings.")

    params = {
        "mode": mode,
        "encrypt": encrypt,
        "key": key,
        "attacks": {
            "gaussian": gauss_sigma if gauss_enable else None,
            "jpeg": int(jpeg_quality) if jpeg_enable else None,
        },
        "weights": (0.5, 0.5),  # w_psnr, w_nc
    }

    # convert watermark PIL to numpy grayscale
    wm_arr = img_to_gray_array(wm_pil)

    # run PSO (parallel)
    best_alpha, best_score, history = pso_optimize_alpha_parallel(
        cover_rgb,
        wm_arr,
        params,
        n_particles=int(n_particles),
        n_iter=int(n_iter),
        alpha_bounds=(alpha_min, alpha_max),
    )

    st.success(f"PSO finished — best alpha = {best_alpha:.4f}, score = {best_score:.4f}")

    # embed final (without attacks to produce final watermarked image)
    watermarked, mask, wm_resized = embed_watermark(cover_rgb, wm_arr, best_alpha, mode=mode, encrypt=encrypt, key=key)

    # extract without attacks for visual quality of extraction (but we will also show attacked extraction)
    extracted_clean = extract_watermark(watermarked, cover_rgb, wm_resized.shape, mode=mode, mask=mask if encrypt else None)

    # Show side-by-side images
    col1, col2, col3 = st.columns(3)
    col1.image(cover_pil, caption="Original Cover", use_column_width=True)
    col2.image(Image.fromarray(watermarked), caption="Watermarked (final)", use_column_width=True)
    col3.image(Image.fromarray(extracted_clean), caption="Extracted watermark (no attack)", use_column_width=True)

    # Evaluate PSNR between cover and watermarked
    gray_cover = cv2.cvtColor(cover_rgb, cv2.COLOR_RGB2GRAY)
    gray_water = cv2.cvtColor(watermarked, cv2.COLOR_RGB2GRAY)
    p = utils_psnr(gray_cover, gray_water)
    st.write(f"PSNR (cover vs watermarked): **{p:.2f} dB**")

    # Show PSO convergence plot
    its = [h[0] for h in history]
    gvals = [h[2] for h in history]
    plt.figure(figsize=(6, 3))
    plt.plot(its, gvals, marker="o")
    plt.xlabel("Iteration")
    plt.ylabel("Global best score")
    plt.title("PSO convergence")
    st.pyplot(plt)

    # Attack testing panel (user can try different attack severities to see extraction robustness)
    st.subheader("Test extraction under attacks (try values and run)")
    tcol1, tcol2, tcol3 = st.columns(3)
    test_gauss = tcol1.slider("Gaussian sigma (test)", 0.0, 50.0, 5.0)
    test_jpeg_q = tcol2.slider("JPEG quality (test)", 1, 100, 60)
    # checkboxes on same column moved to new row to avoid layout quirks
    test_apply_gauss = st.checkbox("Apply Gaussian", value=True)
    test_apply_jpeg = st.checkbox("Apply JPEG", value=True)

    attacked_img = watermarked.copy()
    if test_apply_gauss:
        attacked_img = apply_gaussian_noise(attacked_img, test_gauss)
    if test_apply_jpeg:
        attacked_img = apply_jpeg_compression(attacked_img, int(test_jpeg_q))

    extracted_attacked = extract_watermark(attacked_img, cover_rgb, wm_resized.shape, mode=mode, mask=mask if encrypt else None)
    st.image(attacked_img, caption="Attacked watermarked image", use_column_width=True)
    st.image(extracted_attacked, caption="Extracted watermark (after attack)", use_column_width=True)

    # Compute NC between original resized and attacked-extracted
    orig_bin = (wm_resized > 127).astype(np.float32)
    ex_bin = (extracted_attacked > 127).astype(np.float32)
    nc_val = normalized_correlation(orig_bin, ex_bin)
    st.write(f"Normalized Correlation (NC) between original watermark and extracted (attacked): **{nc_val:.4f}**")

    # Download buttons
    buf = BytesIO()
    Image.fromarray(watermarked).save(buf, format="PNG")
    buf.seek(0)
    st.download_button("Download watermarked PNG", data=buf, file_name="watermarked.png", mime="image/png")

    buf2 = BytesIO()
    Image.fromarray(extracted_attacked).save(buf2, format="PNG")
    buf2.seek(0)
    st.download_button("Download extracted watermark (attacked) PNG", data=buf2, file_name="extracted_attacked.png", mime="image/png")

    # show metadata
    st.subheader("Run metadata")
    st.write(
        {
            "best_alpha": float(best_alpha),
            "pscore": float(best_score),
            "encrypt": encrypt,
            "mode": mode,
            "attacks_used_in_fitness": params["attacks"],
            "particles": int(n_particles),
            "iterations": int(n_iter),
        }
    )