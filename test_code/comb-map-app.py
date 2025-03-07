import os
import json
import time
import base64
import re
from openai import OpenAI
import gradio as gr
import pytesseract
import fitz
import tiktoken
from mimetypes import guess_type
from PIL import Image
from unstructured.partition.pdf import partition_pdf

# --------------------------
# Setup for GPT-4o Pipeline (OpenAI)
# --------------------------
client  = OpenAI(api_key = os.getenv("OPENAI_API_KEY"))
CHUNKS_CACHE_FILE = "comb_pci_processed_chunks.json"
TOKEN_THRESHOLD = 10000  # maximum tokens per chunk allowed

def partition_pdf_by_title(pdf_path):
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

def count_tokens(text: str, model="gpt-4o-mini"):
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def split_text_into_chunks(text: str, max_tokens: int = TOKEN_THRESHOLD, overlap: int = 100) -> list:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = encoding.decode(tokens[start:end])
        chunks.append(chunk)
        start = end - overlap
    return chunks

def process_pdf_sections(pdf_path):
    raw_sections = partition_pdf_by_title(pdf_path)
    processed_sections = {}
    for title, content in raw_sections.items():
        full_section = f"{title}\n{content}"
        tokens = count_tokens(full_section)
        if tokens > TOKEN_THRESHOLD:
            print(f"Section '{title}' is very large ({tokens} tokens); splitting further.")
            sub_chunks = split_text_into_chunks(full_section, max_tokens=TOKEN_THRESHOLD, overlap=100)
            processed_sections[title] = sub_chunks
        else:
            processed_sections[title] = [full_section]
    return processed_sections

def load_or_process_pdf(pdf_path):
    if os.path.exists(CHUNKS_CACHE_FILE):
        with open(CHUNKS_CACHE_FILE, "r", encoding="utf-8") as f:
            processed_sections = json.load(f)
        print("Loaded cached processed PDF sections.")
    else:
        processed_sections = process_pdf_sections(pdf_path)
        with open(CHUNKS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(processed_sections, f, ensure_ascii=False, indent=2)
        print("Processed PDF and saved cache.")
    return processed_sections

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

def check_chunk_relevance(chunk_text: str, image_query: str) -> bool:
    system_message = "You are an expert in PCI-DSS compliance."
    prompt = f"""
    Below is a section from the PCI-DSS Report on Compliance Template:

    \"\"\"{chunk_text}\"\"\"

    Also, here is text extracted from a client's screenshot:
    \"\"\"{image_query}\"\"\"

    Is this section relevant for mapping the client's network/security details to a PCI-DSS control?
    Answer only "Yes" or "No".
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=10,
            temperature=0
        )
        answer = response.choices[0].message.content.strip().lower()
        return answer.startswith("yes")
    except Exception as e:
        print(f"Relevance check error: {e}")
        return False

def map_chunk_to_control(chunk_text: str, image_query: str, image_path: str) -> dict:
    system_message = "You are an expert in PCI-DSS compliance."
    prompt = f"""
    Below is a section from the PCI-DSS Report on Compliance Template:

    \"\"\"{chunk_text}\"\"\"

    And here is text extracted from a client's screenshot:
    \"\"\"{image_query}\"\"\"

    Based on the above, identify the specific control requirement addressed by the screenshot.
    Return your answer as a JSON object with the keys:
    "control_code": a string (e.g., "Requirement 8.2.1.a"),
    "description": a short excerpt from the PDF that describes the requirement,
    "explanation": a brief explanation of why the image satisfies this requirement.
    Output only the JSON with no extra text.
    """
    image_data_url = image_to_data_url(image_path)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=200,
            temperature=0.2
        )
        output = response.choices[0].message.content.strip()
        if not output:
            print("Empty mapping output.")
            return {}
        mapping = json.loads(output)
        if mapping.get("control_code"):
            return mapping
        else:
            return {}
    except Exception as e:
        print(f"Mapping error for chunk: {e}")
        return {}

def aggregate_mappings(mappings: list) -> dict:
    aggregated = {}
    for mapping in mappings:
        code = mapping.get("control_code")
        if not code:
            continue
        if code in aggregated:
            existing_expl = aggregated[code]["explanation"]
            new_expl = mapping.get("explanation", "")
            if new_expl and new_expl not in existing_expl:
                aggregated[code]["explanation"] += " " + new_expl
        else:
            aggregated[code] = {
                "description": mapping.get("description", ""),
                "explanation": mapping.get("explanation", "")
            }
    return aggregated

def process_screenshot_gpt4o(image_file, pdf_file):
    if pdf_file is not None:
        pdf_path = pdf_file.name
        yield "Processing uploaded PCI-DSS ROC PDF..."
        processed_sections = process_pdf_sections(pdf_path)
        with open(CHUNKS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(processed_sections, f, ensure_ascii=False, indent=2)
    else:
        if os.path.exists(CHUNKS_CACHE_FILE):
            yield "Loading cached PCI-DSS ROC PDF..."
            processed_sections = load_or_process_pdf("dummy")
        else:
            yield "No cached PCI-DSS ROC PDF available. Please upload the PCI-DSS ROC PDF."
            return

    yield "Found sections: " + ", ".join(processed_sections.keys())
    yield "Extracting OCR text from image..."
    image_query = ocr_image(image_file)

    mapping_outputs = []
    total_chunks = sum(len(chunks) for chunks in processed_sections.values())
    processed_chunks = 0
    for title, chunk_list in processed_sections.items():
        for chunk in chunk_list:
            processed_chunks += 1
            if check_chunk_relevance(chunk, image_query):
                yield f"Mapping control for chunk {processed_chunks}/{total_chunks} from section '{title}'..."
                mapping = map_chunk_to_control(chunk, image_query, image_file)
                if mapping and mapping.get("control_code"):
                    mapping_outputs.append(mapping)
            yield f"Processed chunk {processed_chunks}/{total_chunks}"
    if not mapping_outputs:
        yield "No mapped controls found for the given image."
        return
    aggregated = aggregate_mappings(mapping_outputs)
    output_lines = ["# Mapped Controls (GPT-4o):"]
    for code, details in aggregated.items():
        output_lines.append(f"- **Control:** {code}")
        output_lines.append(f"  - **Description:** {details['description']}")
        output_lines.append(f"  - **Explanation:** {details['explanation']}\n")
    final_output = "\n".join(output_lines)
    yield final_output

# --------------------------
# Setup for Open Source Pipeline
# --------------------------
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import DonutProcessor, VisionEncoderDecoderModel
import easyocr
import ollama
from collections import defaultdict
from functools import lru_cache

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DB_PATH = "comb_pci_dss_db"
COLLECTION_NAME = "requirements"
EXTRACTION_MODEL = "naver-clova-ix/donut-base-finetuned-docvqa"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def process_and_store_pdf_open_source(pdf_path):
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        return "Document already processed!"
    except Exception:
        collection = chroma_client.create_collection(COLLECTION_NAME)
    
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        page_text = page.get_text("text")
        if not page_text.strip():
            page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_IMAGES)
        full_text += page_text + "\n"
    
    requirement_pattern = re.compile(
        r'(?:^|\n)(\d+\.\d+(?:\.\d+[a-z]?)?)\s*(.*?)\n(.*?)(?=\n\s*\d+\.\d|\Z)',
        flags=re.DOTALL
    )
    matches = requirement_pattern.findall(full_text)
    if not matches:
        return "Error: No requirements found - check document format"
    
    requirements = []
    code_counter = {}
    for match in matches:
        code = match[0].strip()
        title = match[1].strip()
        description = match[2].strip()
        if not code:
            continue
        count = code_counter.get(code, 0) + 1
        code_counter[code] = count
        unique_code = f"{code}_{count}" if count > 1 else code
        requirements.append({
            "original_code": code,
            "unique_code": unique_code,
            "title": title,
            "description": description,
            "full_text": f"{code} {title}\n{description}"
        })
    if not requirements:
        return "Error: No valid requirements extracted"
    model = get_embedding_model()
    embeddings = model.encode(
        [req["full_text"] for req in requirements],
        show_progress_bar=False,
        convert_to_numpy=True
    )
    if len(embeddings) == 0 or embeddings.shape[0] != len(requirements):
        return f"Error: Embedding generation failed ({len(embeddings)} vs {len(requirements)})"
    collection.add(
        ids=[req["unique_code"] for req in requirements],
        embeddings=embeddings.tolist(),
        documents=[req["full_text"] for req in requirements],
        metadatas=[{
            "original_code": req["original_code"],
            "title": req["title"]
        } for req in requirements]
    )
    try:
        verify_count = collection.count()
        if verify_count != len(requirements):
            return f"Error: Storage failed ({verify_count}/{len(requirements)} saved)"
    except Exception as e:
        return f"Verification failed: {str(e)}"
    return f"Processed {len(requirements)} requirements ({len(code_counter)} unique codes)"

class ImageAnalyzer:
    def __init__(self):
        self.processor = DonutProcessor.from_pretrained(EXTRACTION_MODEL)
        self.doc_model = VisionEncoderDecoderModel.from_pretrained(EXTRACTION_MODEL)
        self.doc_model.eval()
        self.general_ocr = easyocr.Reader(['en'], gpu=False)
        try:
            chroma_client = chromadb.PersistentClient(path=DB_PATH)
            self.collection = chroma_client.get_collection(COLLECTION_NAME)
        except Exception as e:
            print(f"ChromaDB error: {e}")
            self.collection = None

    def analyze_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024))
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        task_prompt = "<s_docvqa><s_question>{What technical requirements are shown here?}</s_question><s_answer>"
        decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        output = self.doc_model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
        doc_text = self.processor.batch_decode(output)[0]
        doc_text = re.sub(r"<.*?>", "", doc_text)
        try:
            general_text = self.general_ocr.readtext(image_path, detail=0)
        except Exception as e:
            general_text = [f"OCR Error: {str(e)}"]
        combined_text = f"Document Analysis: {doc_text}\nGeneral OCR: {' '.join(general_text)}"
        response = ollama.generate(
            model="llava:7b-v1.6-mistral-q4_K_M",
            prompt="Describe this technical image in detail for compliance analysis:",
            images=[image_path]
        )
        visual_description = response.get('response', '').strip()
        return f"""Image Analysis:
- Document Understanding: {doc_text}
- OCR Text: {', '.join(general_text)}
- Visual Description: {visual_description}"""

class ComplianceMapper:
    def __init__(self):
        try:
            self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
        except Exception as e:
            print(f"Collection initialization failed: {e}")
            self.collection = None

    def map_requirements(self, image_analysis):
        if not self.collection:
            return {"error": "PCI document not processed. Complete Step 1 first."}
        try:
            model = get_embedding_model()
            query_embedding = model.encode(image_analysis, show_progress_bar=False, convert_to_tensor=True).cpu().numpy().tolist()
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=15,
                include=["metadatas", "documents", "distances"],
                where={"original_code": {"$ne": ""}}
            )
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}
        if not results.get("metadatas") or not results.get("documents"):
            return {"error": "No matching requirements found in document"}
        from collections import defaultdict
        grouped_results = defaultdict(list)
        for metadata, doc, score in zip(results["metadatas"][0], results["documents"][0], results["distances"][0]):
            original_code = metadata.get("original_code", "")
            if not original_code:
                continue
            similarity = 1 - score
            grouped_results[original_code].append({
                "text": doc,
                "title": metadata.get("title", "No title"),
                "score": similarity
            })
        final_results = []
        for code, sections in grouped_results.items():
            avg_score = sum(s["score"] for s in sections) / len(sections)
            combined_text = "\n".join([s["text"] for s in sorted(sections, key=lambda x: x["score"], reverse=True)][:3])[:3000]
            prompt = f"""Analyze PCI-DSS compliance for network infrastructure.

Requirement {code}:
{combined_text}

Observed Configuration:
{image_analysis}

Does the image demonstrate compliance with requirement {code}?
Answer STRICTLY in this format:
- [COMPLIANT/NON-COMPLIANT]:
- Reason: [Technical explanation]"""
            try:
                response = ollama.generate(
                    model="phi3:mini",
                    prompt=prompt,
                    options={'temperature': 0.1}
                )
                response_text = response.get('response', '').strip()
                if "COMPLIANT" in response_text:
                    reasoning = response_text.split("- Reason:", 1)[-1].strip()
                    final_results.append({
                        "code": code,
                        "description": combined_text[:500],
                        "explanation": reasoning,
                        "confidence": f"{avg_score:.0%}"
                    })
            except Exception as e:
                print(f"Error processing {code}: {str(e)}")
        return sorted(final_results, key=lambda x: x["confidence"], reverse=True) if final_results else []

def process_screenshot_open_source(image_file, pdf_file):
    if pdf_file is not None:
        pdf_path = pdf_file.name
        status_pdf = process_and_store_pdf_open_source(pdf_path)
    else:
        if os.path.exists(CHUNKS_CACHE_FILE):
            status_pdf = "Loaded cached PCI-DSS document."
        else:
            return {"error": "No processed PCI-DSS document available. Please upload the PCI-DSS ROC PDF."}
    analyzer = ImageAnalyzer()
    mapper = ComplianceMapper()
    analysis = analyzer.analyze_image(image_file)
    output = mapper.map_requirements(analysis)
    return output

# --------------------------
# Unified Gradio App Function with Two Submit Buttons and Two Output Columns
# --------------------------
def run_gpt4o_pipeline(image_file, pdf_file):
    progress_messages = []
    for message in process_screenshot_gpt4o(image_file, pdf_file):
        progress_messages.append(message)
    # Join messages and return as Markdown
    return "\n\n".join(progress_messages)

# Reuse process_screenshot_gpt4o from GPT-4o section defined earlier
def process_screenshot_gpt4o(image_file, pdf_file):
    if pdf_file is not None:
        pdf_path = pdf_file.name
        yield "Processing uploaded PCI-DSS ROC PDF..."
        processed_sections = process_pdf_sections(pdf_path)
        with open(CHUNKS_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(processed_sections, f, ensure_ascii=False, indent=2)
    else:
        if os.path.exists(CHUNKS_CACHE_FILE):
            yield "Loading cached PCI-DSS ROC PDF..."
            processed_sections = load_or_process_pdf("dummy")
        else:
            yield "No cached PCI-DSS ROC PDF available. Please upload the PCI-DSS ROC PDF."
            return
    yield "Found sections: " + ", ".join(processed_sections.keys())
    yield "Extracting OCR text from image..."
    image_query = ocr_image(image_file)
    mapping_outputs = []
    total_chunks = sum(len(chunks) for chunks in processed_sections.values())
    processed_chunks = 0
    for title, chunk_list in processed_sections.items():
        for chunk in chunk_list:
            processed_chunks += 1
            if check_chunk_relevance(chunk, image_query):
                yield f"Mapping control for chunk {processed_chunks}/{total_chunks} from section '{title}'..."
                mapping = map_chunk_to_control(chunk, image_query, image_file)
                if mapping and mapping.get("control_code"):
                    mapping_outputs.append(mapping)
            yield f"Processed chunk {processed_chunks}/{total_chunks}"
    if not mapping_outputs:
        yield "No mapped controls found for the given image."
        return
    aggregated = aggregate_mappings(mapping_outputs)
    output_lines = ["# Mapped Controls (GPT-4o):"]
    for code, details in aggregated.items():
        output_lines.append(f"- **Control:** {code}")
        output_lines.append(f"  - **Description:** {details['description']}")
        output_lines.append(f"  - **Explanation:** {details['explanation']}\n")
    yield "\n".join(output_lines)

# --------------------------
# Build Gradio Interface with Two Columns
# --------------------------
with gr.Blocks(title="PCI-DSS Compliance Mapping Comparison") as app:
    gr.Markdown("## PCI-DSS Compliance Mapping\nUpload a screenshot image and optionally the PCI-DSS ROC PDF document. Then run each pipeline to compare outputs.")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### GPT-4o Pipeline")
            gpt4o_image = gr.Image(type="filepath", label="Upload Screenshot Image")
            gpt4o_pdf = gr.File(file_types=[".pdf"], label="Upload PCI-DSS ROC PDF (optional)")
            gpt4o_button = gr.Button("Run GPT-4o Pipeline")
            gpt4o_output = gr.Markdown(label="GPT-4o Output")
        with gr.Column():
            gr.Markdown("### Open Source Pipeline")
            os_image = gr.Image(type="filepath", label="Upload Screenshot Image")
            os_pdf = gr.File(file_types=[".pdf"], label="Upload PCI-DSS ROC PDF (optional)")
            os_button = gr.Button("Run Open Source Pipeline")
            os_output = gr.Markdown(label="Open Source Output")
            
    gpt4o_button.click(
        fn=run_gpt4o_pipeline,
        inputs=[gpt4o_image, gpt4o_pdf],
        outputs=gpt4o_output,
        show_progress=True
    )
    
    os_button.click(
        fn=process_screenshot_open_source,
        inputs=[os_image, os_pdf],
        outputs=os_output
    )

if __name__ == "__main__":
    app.launch()
