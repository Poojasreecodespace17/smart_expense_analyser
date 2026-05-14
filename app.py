import streamlit as st
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import fitz
import cv2
import pytesseract
import re

from utils import analyze_expense_ml, save_expense, extract_rows, _preprocess

st.set_page_config(page_title="Smart Expense Analyzer", layout="wide")
st.title("Smart Expense Analyzer")

uploaded_file = st.file_uploader(
    "Upload Receipt (Image or PDF)",
    type=["jpg", "jpeg", "png", "pdf"]
)


# ── helpers ───────────────────────────────────────────────────────────────────

def pil_to_cv2(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img.convert("RGB")), cv2.COLOR_RGB2BGR)


def rows_from_pdf_text(direct_text: str) -> list[str]:
    return [ln.strip() for ln in direct_text.splitlines() if ln.strip()]


def rows_from_cv2(img_cv2: np.ndarray) -> list[str]:
    return extract_rows(img_cv2)


# ── main flow ─────────────────────────────────────────────────────────────────

if uploaded_file:
    file_type = uploaded_file.type
    rows: list[str] = []
    img_cv2 = None

    # ── PDF ───────────────────────────────────────────────────────────────────
    if file_type == "application/pdf":
        file_bytes = uploaded_file.read()
        pdf = fitz.open(stream=file_bytes, filetype="pdf")

        direct_text = "".join(page.get_text() for page in pdf)

        if direct_text.strip():
            # Searchable PDF — use embedded text directly (most accurate)
            rows = rows_from_pdf_text(direct_text)
            st.info("Extracted text directly from PDF (no OCR needed).")
        else:
            st.warning("Scanned PDF — running OCR on first page…")
            page = pdf[0]
            pix   = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            pil   = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            img_cv2 = pil_to_cv2(pil)
            st.image(pil, caption="PDF rendered at 2×")
            with st.spinner("Running OCR…"):
                rows = rows_from_cv2(img_cv2)

    # ── Image ─────────────────────────────────────────────────────────────────
    else:
        pil = Image.open(uploaded_file)
        img_cv2 = pil_to_cv2(pil)
        st.image(pil, caption="Uploaded Receipt")
        with st.spinner("Running OCR…"):
            rows = rows_from_cv2(img_cv2)

    # ── Row display ───────────────────────────────────────────────────────────
    st.subheader("Extracted rows")
    if rows:
        st.dataframe(
            pd.DataFrame({"Row": range(1, len(rows) + 1), "Text": rows}),
            use_container_width=True, hide_index=True
        )
    else:
        st.warning("No text could be extracted.")

    # ── Analysis ──────────────────────────────────────────────────────────────
    text = "\n".join(rows)
    if text.strip():
        with st.spinner("Classifying items…"):
            result = analyze_expense_ml(text)

        items = result.get("items", [])

        if items:
            st.subheader("Classified items")
            items_df = pd.DataFrame(items)
            st.dataframe(items_df, use_container_width=True, hide_index=True)
            st.success(f"Total: ₹{result.get('total', 0):.2f}")

            save_expense(items)

            # Pie chart
            cat_sum = items_df.groupby("category")["price"].sum().sort_values(ascending=False)
            fig, ax = plt.subplots(figsize=(6, 6))
            cat_sum.plot(kind="pie", autopct="%1.1f%%", ax=ax, startangle=140)
            ax.set_ylabel("")
            ax.set_title("Spending by category")
            st.subheader("Category breakdown")
            st.pyplot(fig)
        else:
            st.warning("No purchasable items found. Check the extracted rows above.")

# ── Historical summary ────────────────────────────────────────────────────────
st.divider()
st.subheader("Overall expense history")
try:
    hist = pd.read_csv("expenses.csv")
    # Drop obviously bad rows (names shorter than 4 chars, prices <= 0)
    hist = hist[hist["name"].str.len() >= 4]
    hist = hist[hist["price"] > 0]

    col1, col2 = st.columns(2)
    with col1:
        st.write("**By category**")
        st.dataframe(
            hist.groupby("category")["price"]
                .agg(["sum", "count"])
                .rename(columns={"sum": "Total (₹)", "count": "Items"})
                .sort_values("Total (₹)", ascending=False),
            use_container_width=True
        )
    with col2:
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        hist.groupby("category")["price"].sum().plot(kind="pie", autopct="%1.1f%%", ax=ax2)
        ax2.set_ylabel("")
        st.pyplot(fig2)
except FileNotFoundError:
    st.info("No expense history yet. Upload a receipt to get started.")
except Exception as e:
    st.error(f"Could not load history: {e}")