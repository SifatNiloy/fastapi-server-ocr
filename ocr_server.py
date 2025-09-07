# ocr_server.py
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from PIL import Image
import fitz  # PyMuPDF
import pytesseract
import easyocr

app = FastAPI(title="OCR Server", version="1.1.0")

# Initialize EasyOCR once (English + Bengali)
EASYOCR_LANGS = ["en", "bn"]
reader = easyocr.Reader(EASYOCR_LANGS)

# Accept common types (multer sometimes sends PDFs as octet-stream)
ALLOWED_TYPES = {"image/jpeg", "image/png", "application/pdf", "application/octet-stream"}

# Absolute safety cap to protect this server (in MB). 0 => no cap.
ABSOLUTE_MAX_FILE_MB = 100


# ----------------------------
# Helpers
# ----------------------------
def is_pdf(upload: UploadFile, saved: Path) -> bool:
    """Detect PDFs by content-type + filename extension."""
    return (
        upload.content_type in {"application/pdf", "application/octet-stream"}
        and saved.suffix.lower() == ".pdf"
    )


def check_file_size_limit(size_bytes: int, node_limit_mb: int) -> None:
    """Raise HTTP 413 if over absolute or node-provided limit."""
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
    """Return page count; raise HTTP 422 if it exceeds node-provided limit."""
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
    """Extract selectable (non-OCR) text from PDF."""
    doc = fitz.open(pdf_path)
    try:
        out = []
        for i in range(doc.page_count):
            page = doc.load_page(i)
            out.append(page.get_text("text"))
        return "\n".join(out)
    finally:
        doc.close()


def tesseract_from_pdf(pdf_path: Path, scale: float = 1.0) -> str:
    """OCR a PDF by rasterizing each page at ~100 DPI (scale≈1)."""
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
    """OCR a PDF with EasyOCR, processing all pages."""
    doc = fitz.open(pdf_path)
    try:
        parts = []
        mat = fitz.Matrix(scale, scale)
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


def try_llama_index(input_file: Path) -> Optional[str]:
    """Optional: use LlamaIndex if installed; otherwise skip quietly."""
    try:
        from llama_index.core import SimpleDirectoryReader
    except Exception as e:
        print("LlamaIndex not available:", e)
        return None

    try:
        docs = SimpleDirectoryReader(input_files=[input_file]).load_data()
        text = "\n\n".join(d.text for d in docs)
        return text or ""
    except Exception as e:
        print("LlamaIndex extraction failed:", e)
        return None


# ----------------------------
# API
# ----------------------------
@app.post("/extract-text/")
async def extract_text_from_doc(
    file: UploadFile = File(...),
    max_pages: int = Form(0),    # provided by Node; 0 => no page limit
    max_file_mb: int = Form(0),  # provided by Node; 0 => no size limit
):
    """
    Extract text with this order (first non-empty wins):
      PDF -> native text
      LlamaIndex (optional)
      Tesseract (~300 DPI)
      EasyOCR (all pages)

    Only file size and page count are enforced.
    No 'meaningfulness' checks here.
    """
    # Type gate
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    # Size gate (without loading file into memory)
    file.file.seek(0, os.SEEK_END)
    size_bytes = file.file.tell()
    file.file.seek(0)
    check_file_size_limit(size_bytes, max_file_mb)

    with TemporaryDirectory() as td_str:
        td = Path(td_str)
        filename = file.filename or "upload.bin"
        saved = td / filename

        # Save upload to disk
        with open(saved, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Page limit (PDF only)
        page_count: Optional[int] = None
        if is_pdf(file, saved):
            page_count = check_page_limit(saved, max_pages)

        # 0) Native text (PDF only)
        if is_pdf(file, saved):
            try:
                native = extract_pdf_text_native(saved)
                if native and native.strip():
                    return {
                        "filename": filename,
                        "extracted_text": native,
                        "engine": "PDFNative",
                        "text_len": len(native),
                        "page_count": page_count,
                    }
                else:
                    print("PDF native returned empty; trying fallbacks.")
            except Exception as e:
                print("PDF native extraction failed:", e)

        # 1) LlamaIndex (optional)
        li_text = try_llama_index(saved)
        if li_text and li_text.strip():
            return {
                "filename": filename,
                "extracted_text": li_text,
                "engine": "LlamaIndex",
                "text_len": len(li_text),
                "page_count": page_count,
            }

        # 2) Tesseract OCR (PDF only)
        try:
            t_text = tesseract_from_pdf(saved, scale=1.0)  # ~100 DPI
            if t_text and t_text.strip():
                return {
                    "filename": filename,
                    "extracted_text": t_text,
                    "engine": "Tesseract",
                    "text_len": len(t_text),
                    "page_count": page_count,
                }
            else:
                print("Tesseract returned empty; trying EasyOCR.")
        except Exception as e:
            print("Tesseract failed:", e)

        # 3) EasyOCR OCR (PDF only)
        try:
            ez_text = easyocr_from_pdf(saved, td, scale=1.0)
            if ez_text and ez_text.strip():
                return {
                    "filename": filename,
                    "extracted_text": ez_text,
                    "engine": "EasyOCR",
                    "text_len": len(ez_text),
                    "page_count": page_count,
                }
            else:
                raise HTTPException(status_code=422, detail="No text extracted by any engine.")
        except HTTPException:
            raise
        except Exception as e:
            print("EasyOCR failed:", e)
            raise HTTPException(status_code=500, detail=f"Error processing file with EasyOCR: {e}")
