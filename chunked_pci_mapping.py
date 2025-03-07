import os
import base64
import pickle
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor
from mimetypes import guess_type
from PIL import Image
import pytesseract
from openai import OpenAI
from unstructured.partition.pdf import partition_pdf
import re

# Configuration
PDF_PATH = "PCI-DSS-ROC-Template.pdf"
CHUNK_CACHE = "pci_chunks.pkl"
MAX_CHUNK_SIZE = 25000  # Optimal for GPT-4o's 128k context
MAX_CONCURRENT_REQUESTS = 5  # Stay within rate limits
# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
client = OpenAI(api_key="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")

def get_pdf_chunks() -> List[Dict]:
    """Extract and cache PDF chunks with Unstructured"""
    if Path(CHUNK_CACHE).exists():
        with open(CHUNK_CACHE, "rb") as f:
            return pickle.load(f)
    
    print("Processing PDF with Unstructured...")
    elements = partition_pdf(
        PDF_PATH,
        strategy="auto",
        chunking_strategy="by_title",
        max_characters=MAX_CHUNK_SIZE,
        combine_text_under_n_chars=2000,
        new_after_n_chars=int(MAX_CHUNK_SIZE * 0.9))
    
    chunks = []
    current_chunk = {"title": "Document Start", "content": ""}
    
    for el in elements:
        if el.category == "Title":
            if current_chunk["content"]:
                chunks.append(current_chunk)
            current_chunk = {"title": el.text, "content": el.text + "\n"}
        else:
            current_chunk["content"] += el.text + "\n"
    
    if current_chunk["content"]:
        chunks.append(current_chunk)
    
    with open(CHUNK_CACHE, "wb") as f:
        pickle.dump(chunks, f)
    
    return chunks

def process_image(image_path: str) -> str:
    """Full processing pipeline for an image"""
    # Single OCR and image processing
    ocr_text = pytesseract.image_to_string(Image.open(image_path))
    image_data = image_to_data_url(image_path)
    
    # Process all chunks in parallel
    chunks = get_pdf_chunks()
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(analyze_chunk, chunk, ocr_text, image_data) 
                   for chunk in chunks]
        results = [f.result() for f in futures if f.result()]
    
    return generate_final_report(results, ocr_text, image_data)

def analyze_chunk(chunk: Dict, ocr_text: str, image_data: str) -> Dict:
    """Analyze one document chunk with GPT-4 Vision"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""
                    PCI-DSS Section: {chunk['title']}
                    Section Content: {chunk['content'][:20000]}
                    
                    Image Context: {ocr_text[:1500]}
                    
                    Identify EACT PCI-DSS controls/requirements from this section
                    that are demonstrated in the image. Respond ONLY with the  
                    control(s)/requirements(s) or 'None'"""},
                    {"type": "image_url", "image_url": {"url": image_data}}
                ]
            }],
            max_tokens=100,
            timeout=15
        )
        return parse_response(response.choices[0].message.content, chunk)
    except Exception as e:
        print(f"Error processing chunk: {e}")
        return None

def parse_response(response: str, chunk: Dict) -> Dict:
    """Extract control numbers from GPT response"""
  
    return {
        "controls": response.split(", ") if response else [],
        "section_title": chunk["title"],
        "section_text": chunk["content"]
    } if response else None

def generate_final_report(results: List[Dict], ocr_text: str, image_data: str) -> str:
    """Generate final compliance report"""
    unique_controls = {r["controls"] for r in results if r is not None}
    print(f"Unique controls: {unique_controls}")
    
    if not unique_controls:
        return "No PCI-DSS controls identified in image"
    
    # Get full context for matched controls
    control_contexts = "\n".join(
        f"Control {c} from {r['section_title']}:\n{r['section_text'][:2000]}"
        for r in results for c in r["controls"]
    )
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"""
                Matched Controls: {', '.join(unique_controls)}
                Document Context: {control_contexts}
                OCR Text: {ocr_text}
                
                Create compliance report with:
                1. Exact control numbers
                2. Requirement text quotes
                3. Image evidence description
                4. Implementation assessment"""},
                {"type": "image_url", "image_url": {"url": image_data}}
            ]
        }],
        max_tokens=600
    )
    return response.choices[0].message.content

def image_to_data_url(image_path: str) -> str:
    """Optimized image encoding"""
    mime_type, _ = guess_type(image_path)
    with open(image_path, "rb") as f:
        return f"data:{mime_type};base64,{base64.b64encode(f.read()).decode()}"

def main():
    image_path = "Connfido Network Diagram.png"
    
    print(f"Processing image: {image_path}")
    report = process_image(image_path)
    
    print("\nCompliance Report:")
    print(report)

if __name__ == "__main__":
    main()