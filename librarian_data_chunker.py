import fitz  # PyMuPDF
import re
import os
import json
import gc
import pytesseract
from PIL import Image
import io
from tqdm import tqdm

# Updated Configuration for subfolder
DATA_DIR = "./Data/input_pdfs"
OUTPUT_DIR = "./Chunks"
# Regex to detect common LaTeX markers and 9th grade problem patterns
PROBLEM_PATTERN = re.compile(r'(\\\[|\\\(|\$\$|\$|Exercise\s+\d+\.\d+|Question\s+\d+|Problem\s+\d+)', re.IGNORECASE)

def is_page_empty(text):
    return len(text.strip()) < 10

def get_ocr_text(page):
    # 150 DPI is safer for 4GB RAM
    pix = page.get_pixmap(dpi=150)
    img_data = pix.tobytes("png")
    img = Image.open(io.BytesIO(img_data))
    return pytesseract.image_to_string(img)

def chunk_pdf(file_path):
    chunks = []
    file_name = os.path.basename(file_path)
    
    try:
        doc = fitz.open(file_path)
        current_chunk = ""
        
        for page in doc:
            text = page.get_text("text")
            
            # Use OCR if the PDF is image-only (common with NCERT scans)
            if is_page_empty(text):
                text = get_ocr_text(page)
            
            lines = text.split('\n')
            for line in lines:
                # If we hit a new problem/LaTeX block, break the chunk
                if PROBLEM_PATTERN.search(line) and len(current_chunk) > 400:
                    chunks.append({
                        "source": file_name,
                        "page": page.number + 1,
                        "content": current_chunk.strip()
                    })
                    current_chunk = line + "\n"
                else:
                    current_chunk += line + "\n"
                
                # Length limit for TTS/Whisper compatibility
                if len(current_chunk) > 1200:
                    chunks.append({
                        "source": file_name,
                        "page": page.number + 1,
                        "content": current_chunk.strip()
                    })
                    current_chunk = ""

        if current_chunk:
            chunks.append({"source": file_name, "content": current_chunk.strip()})
        
        doc.close()
    except Exception as e:
        print(f"\n[!] Error processing {file_name}: {e}")
    
    return chunks

def main():
    if not os.path.exists(OUTPUT_DIR): 
        os.makedirs(OUTPUT_DIR)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Could not find folder {DATA_DIR}")
        print("Please ensure your PDFs are in: Megassis/Data/input_pdfs/")
        return

    pdf_files = [f for f in os.listdir(DATA_DIR) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {DATA_DIR}")
        return

    output_path = os.path.join(OUTPUT_DIR, "megassis_chunks.jsonl")
    print(f"Starting chunking of {len(pdf_files)} files...")

    with open(output_path, 'w', encoding='utf-8') as f:
        # Progress bar for better feedback
        for pdf in tqdm(pdf_files, desc="Processing PDFs"):
            path = os.path.join(DATA_DIR, pdf)
            pdf_chunks = chunk_pdf(path)
            
            for chunk in pdf_chunks:
                f.write(json.dumps(chunk) + '\n')
            
            # Keep memory clean
            gc.collect() 
            
    print(f"\nProcessing complete. Check: {output_path}")

if __name__ == "__main__":
    main()
