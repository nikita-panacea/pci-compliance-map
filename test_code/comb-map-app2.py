import os
import json
import re
import numpy as np
import base64
import time
import fitz
import pytesseract
import tiktoken
import chromadb
import easyocr
import gradio as gr
import ollama
from openai import OpenAI
from PIL import Image
from mimetypes import guess_type
from unstructured.partition.pdf import partition_pdf
from transformers import DonutProcessor, VisionEncoderDecoderModel
from sentence_transformers import SentenceTransformer
from functools import lru_cache

# --------------------------
# Configuration and Constants
# --------------------------
CHUNKS_CACHE_FILE = "comb_pci_processed_chunks.json"
TOKEN_THRESHOLD = 10000
OPENAI_CLIENT = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
OSS_MODEL_NAME = "llava:7b-v1.6-mistral-q4_K_M"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --------------------------
#  PDF Processing
# --------------------------
def partition_pdf_by_title(pdf_path):
    """PDF processing for both models"""
    elements = partition_pdf(filename=pdf_path)
    sections = {}
    current_title = "General"
    sections[current_title] = ""
    for element in elements:
        text = element.text.strip()
        category = getattr(element.metadata, "category", "") if element.metadata else ""
        if category in ["Title", "Header"] or (text and text.isupper() and len(text.split()) < 10):
            current_title = text
            if current_title not in sections:
                sections[current_title] = ""
        else:
            sections[current_title] += "\n" + text
    return sections

def process_and_cache_pdf(pdf_path):
    """Process PDF and cache chunks for both models"""
    sections = partition_pdf_by_title(pdf_path)
    processed = {}
    for title, content in sections.items():
        print('Processing section:', title)
        processed[title] = split_text_if_needed(f"{title}\n{content}")
    
    with open(CHUNKS_CACHE_FILE, "w") as f:
        json.dump(processed, f)
    return processed

def split_text_if_needed(text, max_tokens=TOKEN_THRESHOLD):
    """Split text into chunks if exceeds token limit"""
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return [text]
    
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunks.append(encoding.decode(tokens[start:end]))
        start = end - 100  # Overlap
    return chunks

def load_cached_pdf():
    """Load cached PDF chunks for analysis"""
    with open(CHUNKS_CACHE_FILE, "r") as f:
        return json.load(f)

# --------------------------
# Image Processing
# --------------------------

def ocr_image(image_path):
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

def image_to_data_url(image_path: str) -> str:
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

# --------------------------
# GPT-4o Analysis Pipeline
# --------------------------
class GPT4oAnalyzer:
    def analyze(self, image_path, pdf_path=None):
        chunks = process_and_cache_pdf(pdf_path) if pdf_path else load_cached_pdf()
        image_text = ocr_image(image_path)
        mappings = []
        
        for title, chunk_list in chunks.items():
            for chunk in chunk_list:
                if self.is_relevant(chunk, image_text):
                    mapping = self.map_control(chunk, image_text, image_path)
                    print('Relavent mapping found:', mapping)
                    if mapping:
                        mappings.append(mapping)
        
        return self.format_results(mappings)

    def is_relevant(self, chunk, image_text):
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": "Is this PCI-DSS section relevant to the image evidence? Answer Yes/No"
            }, {
                "role": "user",
                "content": f"Section:\n{chunk}\n\nImage Text:\n{image_text}"
            }],
            max_tokens=10
        )
        return response.choices[0].message.content.strip().lower().startswith("yes")

    def map_control(self, chunk, image_text, image_path):
        image_url = image_to_data_url(image_path)
        response = OPENAI_CLIENT.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Map this PCI section:\n{chunk}\n to image evidence"},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    def format_results(self, mappings):
        return "\n".join([f"- **{m['control_code']}**: {m['explanation']}" for m in mappings])

# --------------------------
# Open Source Analysis Pipeline
# --------------------------
class OpenSourceAnalyzer:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
        self.model = SentenceTransformer(EMBEDDING_MODEL)
        self.llm = ollama
    
    def analyze(self, image_path, pdf_path=None):
        chunks = process_and_cache_pdf(pdf_path) if pdf_path else load_cached_pdf()
        image_analysis = self.analyze_image(image_path)
        results = []
        
        for title, chunk_list in chunks.items():
            for chunk in chunk_list:
                if self.is_relevant(chunk, image_analysis):
                    result = self.map_requirement(chunk, image_analysis)
                    print('Relavent mapping found:', result)
                    if result:
                        results.append(result)
        
        return self.format_results(results)

    def analyze_image(self, image_path):
        text = self.reader.readtext(image_path, detail=0)
        return " ".join(text)

    def is_relevant(self, chunk, image_analysis):
        response = self.llm.generate(
            model=OSS_MODEL_NAME,
            prompt=f"Is this PCI requirement section relevant to the image evidence? Answer Yes/No\n\nRequirement:\n{chunk}\n\nEvidence:\n{image_analysis}",
            format="text"
        )
        return response.strip().lower().startswith("yes")

        # chunk_embed = self.model.encode(chunk)
        # image_embed = self.model.encode(image_analysis)
        # similarity = np.dot(chunk_embed, image_embed)
        # return similarity #> 0.3  # Adjust threshold as needed

    def map_requirement(self, chunk, image_analysis):
        response = self.llm.generate(
            model=OSS_MODEL_NAME,
            prompt=f"Map this PCI requirement:\n{chunk}\n\nTo this evidence:\n{image_analysis}",
            format="json"
        )
        try:
            return json.loads(response['response'])
        except:
            return None

    def format_results(self, results):
        return "\n".join([f"- **{r['code']}**: {r['explanation']}" for r in results if r])

# --------------------------
# Unified Gradio Interface
# --------------------------
with gr.Blocks(title="PCI Compliance Analyzer") as app:
    gr.Markdown("# PCI-DSS Compliance Control Mapper")
    
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="PCI-DSS Document", type="filepath", file_types=[".pdf"])
            img_input = gr.Image(label="Evidence Screenshot", type="filepath")
            process_btn = gr.Button("Process PDF Document")
        
        with gr.Column():
            gpt_output = gr.Markdown(label="GPT-4o Results")
            oss_output = gr.Markdown(label="Open Source Results")
    
    with gr.Row():
        gpt_btn = gr.Button("Analyze with GPT-4o", variant="primary")
        oss_btn = gr.Button("Analyze with Open Source", variant="secondary")

    # Initialize analyzers
    gpt_analyzer = gr.State(GPT4oAnalyzer())
    oss_analyzer = gr.State(OpenSourceAnalyzer())

    # Event handlers
    process_btn.click(
        fn=lambda pdf: process_and_cache_pdf(pdf.name),
        inputs=pdf_input,
        outputs=[]
    )

    gpt_btn.click(
        fn=lambda img, pdf, analyzer: analyzer.analyze(img.name, pdf.name if pdf else None),
        inputs=[img_input, pdf_input, gpt_analyzer],
        outputs=gpt_output
    )

    oss_btn.click(
        fn=lambda img, pdf, analyzer: analyzer.analyze(img.name, pdf.name if pdf else None),
        inputs=[img_input, pdf_input, oss_analyzer],
        outputs=oss_output
    )

if __name__ == "__main__":
    app.launch()