import io
import streamlit as st
import pdfplumber
from PIL import Image
from typing import Optional

def render_pdf_page_as_image(pdf_bytes: bytes, page_number: int) -> Optional[Image.Image]:
    """
    Renders a specific 1-indexed page from a PDF byte buffer into a PIL Image.
    """
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            if 1 <= page_number <= len(pdf.pages):
                page = pdf.pages[page_number - 1]
                page_img = page.to_image(resolution=150).original
                return page_img
    except Exception as e:
        st.error(f"Error rendering PDF page {page_number}: {e}")
    return None

def display_evidence_drawer(pdf_bytes: bytes, page_number: int, line_item: str, reported_value: str):
    """
    Renders an expandable visual audit box displaying the regulatory disclosure page.
    """
    with st.expander(f"🔍 Source Evidence Trace — Page {page_number} ({line_item})", expanded=False):
        st.caption(f"Verifying Reported Metric: **{line_item} = {reported_value}** directly on filed regulatory disclosure.")
        img = render_pdf_page_as_image(pdf_bytes, page_number)
        if img:
            st.image(img, caption=f"Filing Disclosure — Page {page_number}", use_container_width=True)
        else:
            st.info(f"Page image preview unavailable for Page {page_number}. Refer to source document.")