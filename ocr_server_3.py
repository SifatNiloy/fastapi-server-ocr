import shutil
import os
import re
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import pytesseract
import fitz
import easyocr
import numpy as np
from llama_index.core import SimpleDirectoryReader
from pathlib import Path
from tempfile import TemporaryDirectory

# Initialize EasyOCR with the desired languages
# For Bengali, use 'bn'
reader = easyocr.Reader(['en', 'bn'])
app = FastAPI()

# Define validation rules
FILE_SIZE_LIMIT = 10 * 1024 * 1024  # 10 MB
ALLOWED_TYPES = ["image/jpeg", "image/png", "application/pdf"]

# def is_meaningful(text: str) -> bool:
#     """
#     Checks if the extracted text is meaningful (not just gibberish).
#     This function is now more robust and less restrictive.
#     """
#     # Remove leading/trailing whitespace
#     text = text.strip()
    
#     # 1. Check for minimum length of non-whitespace characters
#     if len(text.replace(" ", "")) < 10:
#         return False
        
#     # 2. Check for at least a few words or phrases
#     words = text.split()
#     if len(words) < 2:
#         return False
        
#     # 3. Check for a reasonable ratio of "valid" characters.
#     # We can use a combination of letters, numbers, and common punctuation.
#     # This is more robust than a simple alphanumeric check for multi-lingual content.
#     valid_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?()[]{}\'\"-')
#     # For Bengali, we add the Bengali character set
#     valid_chars.update(set('অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহািীুূৃেৈোৌৎ্ংঃঁৎ'))

#     text_chars = set(text.lower())
#     meaningful_chars = text_chars.intersection(valid_chars)
    
#     if len(text) > 0 and len(meaningful_chars) / len(text_chars) < 0.25:
#         return False
        
#     return True


@app.post("/extract-text/")
async def extract_text_from_doc(file: UploadFile = File(...)):
    """
    Extracts text from an uploaded document with validation and OCR fallback.
    """
    # 1. Validation
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type.")
    
    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)
    if file_size > FILE_SIZE_LIMIT:
        raise HTTPException(status_code=400, detail="File size exceeds 10MB limit.")

    # Use a temporary directory to handle the file
    with TemporaryDirectory() as temp_dir:
        temp_file_path = Path(temp_dir) / file.filename
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 2. Try LlamaIndex (Primary)
        try:
            print("Attempting extraction with LlamaIndex...")
            llama_index_documents = SimpleDirectoryReader(input_files=[temp_file_path]).load_data()
            llama_index_text = "\n\n".join([doc.text for doc in llama_index_documents])
            
            # if is_meaningful(llama_index_text):
            if (llama_index_text):
                print("LlamaIndex extraction successful.")
                return {"filename": file.filename, "extracted_text": llama_index_text, "engine": "LlamaIndex"}
            else:
                print("LlamaIndex text not meaningful. Falling back...")
        except Exception as e:
            print(f"LlamaIndex extraction failed: {e}. Falling back to Tesseract.")

        # 3. Fallback to Tesseract (Secondary)
        tesseract_text = ""
        try:
            # Add lang='ben' for Bengali
            if file.content_type.startswith("image/"):
                img = Image.open(temp_file_path)
                tesseract_text = pytesseract.image_to_string(img, lang='eng+ben')
            elif file.content_type == "application/pdf":
                doc = fitz.open(temp_file_path)
                full_text = []
                for page_num in range(doc.page_count):
                    page = doc.load_page(page_num)
                    pix = page.get_pixmap()
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    full_text.append(pytesseract.image_to_string(img, lang='eng+ben'))
                tesseract_text = "\n".join(full_text)
                doc.close()

            # if is_meaningful(tesseract_text):
            if (tesseract_text):
                print("Tesseract extraction successful.")
                return {"filename": file.filename, "extracted_text": tesseract_text, "engine": "Tesseract"}
            else:
                print("Tesseract text not meaningful. Falling back...")
        except Exception as e:
            print(f"Tesseract extraction failed: {e}. Falling back to EasyOCR.")

        # 4. Fallback to EasyOCR (Tertiary)
        easyocr_text = ""
        try:
            print("Attempting extraction with EasyOCR.")
            if file.content_type.startswith("image/"):
                easyocr_result = reader.readtext(str(temp_file_path))
            else: # PDF, process first page only to save resources
                doc = fitz.open(temp_file_path)
                page = doc.load_page(0)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                temp_img_path = Path(temp_dir) / "temp_page0.png"
                img.save(temp_img_path)
                doc.close()
                easyocr_result = reader.readtext(str(temp_img_path))
            
            easyocr_text = " ".join([text for (bbox, text, prob) in easyocr_result])

            if not easyocr_text:
                raise HTTPException(status_code=400, detail="Could not extract text from document using any OCR engine.")

            print("EasyOCR extraction successful.")
            return {"filename": file.filename, "extracted_text": easyocr_text, "engine": "EasyOCR"}
        except Exception as e:
            print(f"EasyOCR extraction failed: {e}.")
            raise HTTPException(status_code=500, detail=f"Error processing file with EasyOCR: {e}")
        

        