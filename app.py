import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
from io import BytesIO

from train_pso import pso_optimize_alpha
from watermarking import embed_watermark, extract_watermark

st.set_page_config(page_title="Advanced Watermarking + PSO",layout="wide")
st.title("DWT Watermarking with PSO Optimization")

cover_file = st.file_uploader("Upload Cover Image",type=["png","jpg","jpeg"])
watermark_file = st.file_uploader("Upload Watermark",type=["png","jpg","jpeg"])

text_watermark = st.text_input("Or Watermark Text","Demo WM")

mode = st.radio("Mode",("grayscale","color"))

encrypt = st.checkbox("Enable XOR Encryption")
key = st.text_input("Encryption Key","secret") if encrypt else None

n_particles = st.number_input("Particles",4,64,12)
n_iter = st.number_input("Iterations",1,100,20)

run_and_embed = st.button("Run PSO + Embed")

if cover_file is None:
    st.stop()

cover = Image.open(cover_file).convert("L").resize((512,512))
cover_arr = np.array(cover)

if watermark_file:
    wm_pil = Image.open(watermark_file).convert("L")
else:
    img = Image.new("L",(128,128),0)
    draw = ImageDraw.Draw(img)
    draw.text((20,50),text_watermark,255)
    wm_pil = img

wm = wm_pil.resize((128,128))
wm_arr = np.array(wm)

st.image(cover,width=250)
st.image(wm,width=150)

if run_and_embed:

    best_alpha = pso_optimize_alpha(
        cover_arr,
        wm_arr,
        int(n_particles),
        int(n_iter)
    )

    st.success(f"Best Alpha = {best_alpha}")

    watermarked,_ = embed_watermark(
        cover_arr,
        wm_arr,
        best_alpha,
        mode=mode,
        encrypt=encrypt,
        key=key
    )

    extracted = extract_watermark(
        cover_arr,
        watermarked,
        best_alpha,
        encrypt=encrypt,
        key=key
    )

    col1,col2,col3 = st.columns(3)
    col1.image(cover,caption="Original")
    col2.image(watermarked,caption="Watermarked")
    col3.image(extracted,caption="Extracted")

    psnr = cv2.PSNR(cover_arr,watermarked)
    st.write(f"PSNR: {psnr:.2f}")

    buf=BytesIO()
    Image.fromarray(watermarked).save(buf,format="PNG")
    st.download_button("Download Watermarked",
                       buf.getvalue(),
                       "watermarked.png")