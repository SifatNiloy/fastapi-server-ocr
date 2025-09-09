import os
import shutil
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import torch
import fitz  # PyMuPDF
import pytesseract
import easyocr
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks

# ----------------------------
# Logging setup
# ----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ----------------------------
# App Initialization
# ----------------------------
app = FastAPI(title="OCR Server", version="2.0.0")

# Configure OCR engines
EASYOCR_LANGS = ["en", "bn"]

# Lazy load EasyOCR reader to avoid heavy startup cost
reader: Optional[easyocr.Reader] = None


def get_easyocr_reader() -> easyocr.Reader:
    """Return a singleton EasyOCR reader (GPU if available)."""
    global reader
    if reader is None:
        logger.info("Initializing EasyOCR reader (GPU=%s)", torch.cuda.is_available())
        reader = easyocr.Reader(EASYOCR_LANGS, gpu=torch.cuda.is_available())
    return reader


# Allowed file types
ALLOWED_TYPES = {"application/pdf"}
ABSOLUTE_MAX_FILE_MB = 100  # Hard safety cap


# ----------------------------
# Helpers
# ----------------------------
def sanitize_filename(filename: str) -> str:
    """Prevent directory traversal attacks (e.g., ../../etc/passwd)."""
    return Path(filename).name


def check_file_size_limit(size_bytes: int, node_limit_mb: int) -> None:
    """Enforce absolute + node-provided file size limits."""
    if ABSOLUTE_MAX_FILE_MB and size_bytes > ABSOLUTE_MAX_FILE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (> {ABSOLUTE_MAX_FILE_MB} MB absolute server limit).",
        )
    if node_limit_mb and size_bytes > node_limit_mb * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (> {node_limit_mb} MB).",
        )


def check_page_limit(pdf_path: Path, node_max_pages: int) -> int:
    """Return page count; enforce page limit if provided."""
    doc = fitz.open(pdf_path)
    try:
        pages = doc.page_count
    finally:
        doc.close()
    if node_max_pages and pages > node_max_pages:
        raise HTTPException(
            status_code=422,
            detail=f"Page limit exceeded ({pages} > {node_max_pages}).",
        )
    return pages


def extract_pdf_text_native(pdf_path: Path) -> str:
    """Extract selectable text directly from PDF (non-OCR)."""
    doc = fitz.open(pdf_path)
    try:
        out = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            out.append(page.get_text("text"))
        return "\n".join(out)
    finally:
        doc.close()


def try_llama_index(input_file: Path) -> Optional[str]:
    """Try extracting text with LlamaIndex (optional dependency)."""
    try:
        from llama_index.core import SimpleDirectoryReader
    except ImportError:
        logger.warning("LlamaIndex not installed, skipping.")
        return None

    try:
        docs = SimpleDirectoryReader(input_files=[input_file]).load_data()
        return "\n\n".join(d.text for d in docs)
    except Exception as e:
        logger.error("LlamaIndex extraction failed: %s", e)
        return None


def tesseract_from_pdf(pdf_path: Path, scale: float = 1.0) -> str:
    """OCR PDF using Tesseract by rasterizing pages."""
    doc = fitz.open(pdf_path)
    try:
        mat = fitz.Matrix(scale, scale)
        out = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            out.append(pytesseract.image_to_string(img, lang="eng+ben"))
        return "\n".join(out)
    finally:
        doc.close()


def easyocr_from_pdf(pdf_path: Path, workdir: Path, scale: float = 1.0) -> str:
    """OCR PDF using EasyOCR by rasterizing pages."""
    doc = fitz.open(pdf_path)
    try:
        parts = []
        mat = fitz.Matrix(scale, scale)
        reader = get_easyocr_reader()
        for i in range(doc.page_count):
            page = doc.load_page(i)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            tmp = workdir / f"page_{i}.png"
            img.save(tmp)
            res = reader.readtext(str(tmp))
            parts.append(" ".join(t for (_, t, _) in res))
        return "\n".join(parts)
    finally:
        doc.close()


# ----------------------------
# API
# ----------------------------
@app.get("/healthz")
async def health_check():
    """Kubernetes/Docker health probe."""
    return {"status": "ok"}


@app.post("/extract-text/")
async def extract_text_from_doc(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_pages: int = Form(0),    # node-provided limit (0 = no limit)
    max_file_mb: int = Form(0),  # node-provided limit (0 = no limit)
):
    """
    Extract text from PDF in this order:
      1. PDF Native Text
      2. LlamaIndex
      3. Tesseract OCR
      4. EasyOCR
    """

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type (PDF only).")

    # File size check
    file.file.seek(0, os.SEEK_END)
    size_bytes = file.file.tell()
    file.file.seek(0)
    check_file_size_limit(size_bytes, max_file_mb)

    with TemporaryDirectory() as td_str:
        td = Path(td_str)
        filename = sanitize_filename(file.filename or "upload.pdf")
        saved = td / filename

        # Save uploaded file
        with open(saved, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Page limit check
        page_count: Optional[int] = check_page_limit(saved, max_pages)

        # Pipeline
        try:
            # 1. Native text
            native = extract_pdf_text_native(saved)
            if native.strip():
                return {
                    "filename": filename,
                    "engine": "PDFNative",
                    "extracted_text": native,
                    "text_len": len(native),
                    "page_count": page_count,
                }

            # 2. LlamaIndex
            li_text = try_llama_index(saved)
            if li_text and li_text.strip():
                return {
                    "filename": filename,
                    "engine": "LlamaIndex",
                    "extracted_text": li_text,
                    "text_len": len(li_text),
                    "page_count": page_count,
                }

            # 3. Tesseract
            t_text = tesseract_from_pdf(saved, scale=1.0)
            if t_text.strip():
                return {
                    "filename": filename,
                    "engine": "Tesseract",
                    "extracted_text": t_text,
                    "text_len": len(t_text),
                    "page_count": page_count,
                }

            # 4. EasyOCR
            ez_text = easyocr_from_pdf(saved, td, scale=1.0)
            if ez_text.strip():
                return {
                    "filename": filename,
                    "engine": "EasyOCR",
                    "extracted_text": ez_text,
                    "text_len": len(ez_text),
                    "page_count": page_count,
                }

        except HTTPException:
            raise
        except Exception as e:
            logger.exception("OCR pipeline failed")
            raise HTTPException(status_code=500, detail="Internal OCR error")

        raise HTTPException(status_code=422, detail="No text extracted by any engine.")
