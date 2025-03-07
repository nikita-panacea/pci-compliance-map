import os
import json
from openai import OpenAI
import base64
from mimetypes import guess_type
from PIL import Image
import fitz
import pytesseract
import tiktoken
import gradio as gr
from unstructured.partition.pdf import partition_pdf

# Set your OpenAI API key (and organization if needed)
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
# openai.organization = os.getenv("OPENAI_ORG_ID")

# File for caching processed PDF sections and token threshold
CHUNKS_CACHE_FILE = "pci_processed_chunks.json"
TOKEN_THRESHOLD = 10000  # maximum tokens per chunk allowed

# --------------------------
# PDF Partitioning and Caching
# --------------------------
def partition_pdf_by_title(pdf_path):
    """
    Partition the PDF into sections based on titles/headers using unstructured.
    Returns a dict mapping section titles to their full text.
    """
    elements = partition_pdf(filename=pdf_path)
    sections = {}
    current_title = "General"
    sections[current_title] = ""
    for element in elements:
        text = element.text.strip()
        # Safely access metadata (using getattr since it's not a dict)
        category = getattr(element.metadata, "category", "") if element.metadata else ""
        if category in ["Title", "Header"] or (text and text.isupper() and len(text.split()) < 10):
            current_title = text
            if current_title not in sections:
                sections[current_title] = ""
        else:
            sections[current_title] += "\n" + text
    return sections

def count_tokens(text: str, model="gpt-4o-mini"):
    """Count tokens using tiktoken."""
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def split_text_into_chunks(text: str, max_tokens: int = TOKEN_THRESHOLD, overlap: int = 100) -> list:
    """
    Split a long text into overlapping chunks based on token count.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + max_tokens
        chunk = encoding.decode(tokens[start:end])
        chunks.append(chunk)
        start = end - overlap  # maintain overlap for context continuity
    return chunks

def process_pdf_sections(pdf_path):
    """
    Partition the PDF by title and for each section that exceeds TOKEN_THRESHOLD,
    further split it into sub-chunks.
    Returns a dict mapping section titles to lists of text chunks.
    """
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
    """
    If a processed PDF cache exists, load it.
    Otherwise, process the PDF and cache the results.
    """
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

# --------------------------
# Image and OCR Utilities
# --------------------------

def ocr_image(image_path):
    """Extract text from an image using pytesseract."""
    image = Image.open(image_path)
    return pytesseract.image_to_string(image)

def image_to_data_url(image_path: str) -> str:
    """Convert an image file to a base64-encoded data URL."""
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

# --------------------------
# Relevance Check and Mapping per Chunk
# --------------------------

def check_chunk_relevance(chunk_text: str, image_query: str) -> bool:
    """
    Ask GPT-4 if a given chunk is relevant to the client's image.
    Returns True if the answer starts with "yes".
    """
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
    """
    Use GPT-4 Vision to map a relevant chunk to a specific PCI-DSS control.
    Returns a JSON object (parsed as dict) with keys:
      - "control_code": string (e.g., "Requirement 8.2.1.a")
      - "description": excerpt from the PDF chunk
      - "explanation": why the image satisfies this requirement.
    If no mapping is found, returns {}.
    """
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
    """
    Aggregate mapping dictionaries from each relevant chunk into a deduplicated dict keyed by control code.
    For duplicate control codes, merge the explanations.
    """
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

# --------------------------
# Gradio App Function (Generator for Progress)
# --------------------------
def process_screenshot(image_file, pdf_file):
    """
    Gradio function:
    - image_file: screenshot image file (filepath)
    - pdf_file: optional PCI-DSS ROC PDF file.
    Returns final aggregated mapped controls as Markdown.
    Yields progress updates.
    """
    # Step 1: Load or process the PDF
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
    
    # Step 2: Extract OCR from the screenshot
    yield "Extracting OCR text from image..."
    image_query = ocr_image(image_file)
    
    # Step 3: Process each chunk for relevance and mapping
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
        output_lines = []
        output_lines.append("### Mapped Controls:")
        output_lines.append("No mapped controls found for the given image.")
        final_output = "\n".join(output_lines)
        yield final_output
        return
    
    aggregated = aggregate_mappings(mapping_outputs)
    output_lines = []
    output_lines.append("# Mapped Controls:")
    for code, details in aggregated.items():
        output_lines.append(f"- **Control:** {code}")
        output_lines.append(f"  - **Description:** {details['description']}")
        output_lines.append(f"  - **Explanation:** {details['explanation']}\n")
    final_output = "\n".join(output_lines)
    yield final_output

# --------------------------
# Build Gradio Interface
# --------------------------
iface = gr.Interface(
    fn=process_screenshot,
    inputs=[
        gr.Image(type="filepath", label="Upload Screenshot Image"),
        gr.File(file_types=[".pdf"], label="Upload PCI-DSS ROC PDF (optional)")
    ],
    outputs=gr.Markdown(label="Mapped PCI-DSS Controls"),
    title="PCI-DSS Compliance Mapping",
    description="Upload a screenshot of a network/security configuration and optionally the PCI-DSS ROC PDF document. The app processes the PDF (or loads it from cache) and maps the image to the specific control(s)/requirement(s) from the document.",
    allow_flagging="never",
    live=False,
    show_progress=True  # shows progress updates from generator
)

if __name__ == "__main__":
    iface.launch()