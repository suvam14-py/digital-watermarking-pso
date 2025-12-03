# Advanced DWT-DCT Watermarking with PSO (Streamlit)

Features:
- Hybrid DWT + block-DCT watermark embedding.
- PSO optimization (parallelized) for embedding strength alpha.
- Simulated attacks included in fitness: Gaussian noise, JPEG compression.
- Optional XOR encryption for watermark bits.
- Supports grayscale and color (embed into luminance Y) modes.
- Streamlit UI for upload, PSO run, extraction, attack testing, and downloads.

## Run locally
1. Create venv:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install -r requirements.txt
   streamlit run app.py