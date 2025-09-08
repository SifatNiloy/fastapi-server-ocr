import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from concurrent.futures import ThreadPoolExecutor

import fitz  # PyMuPDF for PDF handling
import numpy as np
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import easyocr
from llama_index.core import SimpleDirectoryReader

# -------------------------
# CONFIGURATION
# -------------------------

# Initialize FastAPI app
app = FastAPI()

# File validation rules
FILE_SIZE_LIMIT = 10 * 1024 * 1024  # 10 MB max file size
ALLOWED_TYPES = ["image/jpeg", "image/png", "application/pdf"]

# Initialize EasyOCR reader with GPU enabled
# EasyOCR will use CUDA if available (your RTX 3050)
reader = easyocr.Reader(['en', 'bn'], gpu=True)

# Use thread pool for OCR so it doesn’t block FastAPI
executor = ThreadPoolExecutor(max_workers=4)


# -------------------------
# HELPER FUNCTIONS
# -------------------------

def extract_with_llamaindex(file_path: Path) -> str:
    """Try extracting text using LlamaIndex (works best for PDFs with selectable text)."""
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    return "\n\n".join([doc.text for doc in documents])


def extract_with_tesseract(file_path: Path, file_type: str) -> str:
    """Fallback OCR using Tesseract (CPU only)."""
    if file_type.startswith("image/"):
        img = Image.open(file_path)
        return pytesseract.image_to_string(img, lang="eng+ben")

    elif file_type == "application/pdf":
        doc = fitz.open(file_path)
        texts = []
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            texts.append(pytesseract.image_to_string(img, lang="eng+ben"))
        doc.close()
        return "\n".join(texts)


def extract_with_easyocr(file_path: Path, file_type: str, temp_dir: Path) -> str:
    """Final fallback OCR using EasyOCR (GPU-accelerated)."""
    if file_type.startswith("image/"):
        results = reader.readtext(str(file_path))
    else:  # for PDF, only process first page to save time
        doc = fitz.open(file_path)
        page = doc.load_page(0)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        temp_img_path = temp_dir / "page0.png"
        img.save(temp_img_path)
        doc.close()
        results = reader.readtext(str(temp_img_path))

    return " ".join([text for (_, text, _) in results])


# -------------------------
# API ENDPOINT
# -------------------------

@app.post("/extract-text/")
async def extract_text_from_doc(file: UploadFile = File(...)):
    """
    Extract text from uploaded files (JPG, PNG, PDF).
    Pipeline:
    1. Try LlamaIndex (fastest for PDFs with text).
    2. Fallback to Tesseract (CPU OCR).
    3. Fallback to EasyOCR (GPU OCR, slower but more robust).
    """

    # 1. Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type.")

    # 2. Validate file size
    file.file.seek(0, 2)  # move cursor to end
    size = file.file.tell()
    file.file.seek(0)  # reset cursor
    if size > FILE_SIZE_LIMIT:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit.")

    # 3. Store file in temp dir
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        temp_file_path = temp_dir_path / file.filename

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # 4. Try extraction methods in order (non-blocking using thread pool)
        try:
            text = await app.loop.run_in_executor(executor, extract_with_llamaindex, temp_file_path)
            if text.strip():
                return {"filename": file.filename, "extracted_text": text, "engine": "LlamaIndex"}
        except Exception as e:
            print(f"LlamaIndex failed: {e}")

        try:
            text = await app.loop.run_in_executor(executor, extract_with_tesseract, temp_file_path, file.content_type)
            if text.strip():
                return {"filename": file.filename, "extracted_text": text, "engine": "Tesseract"}
        except Exception as e:
            print(f"Tesseract failed: {e}")

        try:
            text = await app.loop.run_in_executor(executor, extract_with_easyocr, temp_file_path, file.content_type, temp_dir_path)
            if text.strip():
                return {"filename": file.filename, "extracted_text": text, "engine": "EasyOCR"}
        except Exception as e:
            print(f"EasyOCR failed: {e}")

        # If nothing worked
        raise HTTPException(status_code=500, detail="Failed to extract text with all OCR engines.")
