from PIL import Image
import easyocr
import json
import ollama
import os
import re
import time
import base64
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

#-------------------------
# Load and Save JSON files
#-------------------------
def load_json_file(filename):
    """Load JSON data from a file."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_file(data, filename):
    """Save JSON data to a file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# Path to processed PDF sections (JSON file)
pdf_sections_file = "open-source-pipeline/processed_pci_chunks.json"

#------------------
# Image OCR 
#------------------
class OCRProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False) #NOTE: True on Omen
    
    def process_image(self, image_path, output_file="ocr_results.json"):
        print("Extracting text from image...")
        result = self.reader.readtext(image_path, detail=0)
        print("Text extraction complete.")
        with open(output_file, "w") as f:
            json.dump({"text": "\n".join(result)}, f)
        return output_file

#-------------------------
# Open Source Pipeline
#-------------------------

# Image Analysis
class VisualAnalyzer:
    def __init__(self):
        self.model_name = "minicpm-v:latest"  # or "llama3.2-vision:latest"
        
    def analyze(self, image_path, image_ocr, output_file="vision_analysis_minicpm.json"):
        prompt = f"""
        You are an expert in PCI-DSS compliance network security.
        Provide a detailed report on the given image for compliance analysis:
        - List all systems and devices present and their roles.
        - Describe all network connections and architecture.
        - Explain the network topology and security measures.
        - Identify any potential vulnerabilities or security risks.
        Context (from OCR): {image_ocr}
        """
        print("Analyzing image generating report...")
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            images=[image_path]
        )
        print("Analysis complete.")
        with open(output_file, "w") as f:
            json.dump({"analysis": response.get('response', '')}, f)
        return output_file

# Initialize components
processor = OCRProcessor()
analyzer = VisualAnalyzer()

#------------------------------
# Compliance Mapping Functions
#------------------------------
def map_section_to_requirement(section_title, section_text, image_analysis):
    """
    For a given PDF section (with title and text) and a detailed image analysis report,
    sends a prompt to the open-source reasoning model (deepseek-r1:7b via Ollama) to
    determine which requirement controls are satisfied by the image evidence.
    Returns a JSON array of mapping objects. Each mapping object has:
      - "control_code": string
      - "description": string
      - "explanation": string
      - "missing_aspects": string
    If none are found, returns an empty array.
    """
    if len(section_text.strip()) < 10:
        return []
    
    prompt = f"""
    You are an expert in PCI-DSS compliance. Analyze the following section from a PCI-DSS ROC compliance document and the provided detailed image analysis report.
    The document section (including its heading) contains requirement controls, guidelines, and auditor instructions.
    Your task:
    1. Read and understand the given section carefully.
    2. Determine all specific requirement/control(s) from the section that are satisfied by the image evidence.
    3. For each one, provide the exact control/requirement code (e.g., "Requirement 8.2.1.a").
    4. Provide a short excerpt from the section that describes that requirement.
    5. Explain briefly why the image evidence satisfies this requirement.
    6. Note if any aspects of the requirement are not fully satisfied.
    Always return your answer as a JSON array of objects with exactly these keys:
    "control_code": string,
    "description": string,
    "explanation": string,
    "missing_aspects": string
    Output only the JSON array with no extra text.
    Do not change the output format.
    Do not change the JSON key names.
    Do not change the data type of JSON objects, always return a string.
    For "missing_aspects", always return a string "",  do not return null or list[].

    Section Title: {section_title}
    Section Content:
    {section_text}

    Image Analysis Report:
    {image_analysis}
    """
    try:
        response = ollama.generate(
            model="deepseek-r1:7b",
            prompt=prompt,
            options={'temperature': 0.1}
        )
        output = response.get("response", "").strip()
        if not output:
            print(f"[{section_title}] Empty output from model.")
            return []
        json_match = re.search(r"\[.*\]", output, re.DOTALL)
        if not json_match:
            print(f"[{section_title}] No JSON array found in output.")
            return []
        json_str = json_match.group(0)
        mapping_list = json.loads(json_str)
        if isinstance(mapping_list, list):
            return [m for m in mapping_list if m.get("control_code")]
        else:
            return []
    except Exception as e:
        print(f"[{section_title}] Mapping error: {e}")
        return []

def aggregate_mappings(mapping_list):
    """
    Aggregate mapping objects from multiple sections into a deduplicated dictionary keyed by control code.
    Merge explanations and missing_aspects for duplicate control codes.
    """
    aggregated = {}
    for mapping in mapping_list:
        code = mapping.get("control_code")
        if not code:
            continue
        if code in aggregated:
            existing_expl = aggregated[code]["explanation"]
            new_expl = mapping.get("explanation", "")
            if new_expl and new_expl not in existing_expl:
                aggregated[code]["explanation"] += " " + new_expl
            existing_missing = aggregated[code]["missing_aspects"]
            new_missing = mapping.get("missing_aspects", "")
            if new_missing and new_missing not in existing_missing:
                aggregated[code]["missing_aspects"] += " " + new_missing
        else:
            aggregated[code] = {
                "description": mapping.get("description", ""),
                "explanation": mapping.get("explanation", ""),
                "missing_aspects": mapping.get("missing_aspects", "")
            }
    return aggregated

def format_mappings_to_markdown(aggregated_mappings):
    """Convert aggregated mappings to formatted Markdown"""
    markdown_output = ""
    for section_title, controls in aggregated_mappings.items():
        markdown_output += f"## {section_title}\n\n"
        for control in controls:
            markdown_output += f"**Control Code**: {control['control_code']}\n\n"
            markdown_output += f"**Description**: {control['description']}\n\n"
            markdown_output += f"**Explanation**: {control['explanation']}\n\n"
            if control['missing_aspects'].strip():
                markdown_output += f"**Missing Aspects**: {control['missing_aspects']}\n\n"
            markdown_output += "---\n"
    return markdown_output

#-------------------------
# Processing Functions
#-------------------------
def process_image_and_extract_ocr(image_path):
    """Handle image upload and OCR processing"""
    if not image_path:
        raise gr.Error("Please upload an image first")
    
    ocr_result_path = processor.process_image(image_path)
    with open(ocr_result_path, 'r') as f:
        ocr_data = json.load(f)
    return ocr_data['text']

def run_open_source_pipeline(image_path, ocr_text):
    """Handle open-source pipeline processing"""
    try:
        analysis_report_path = analyzer.analyze(image_path, ocr_text)
        with open(analysis_report_path, 'r') as f:
            analysis_data = json.load(f)
        image_analysis = analysis_data['analysis']
        
        sections = load_json_file(pdf_sections_file)
        
        all_mappings = {}
        for title, text in sections.items():
            mappings = map_section_to_requirement(title, text, image_analysis)
            if mappings:
                all_mappings[title] = mappings
        
        aggregated = aggregate_mappings(all_mappings)
        markdown_output = format_mappings_to_markdown(all_mappings)
        save_json_file(all_mappings, "open_source_mappings.json")
        return markdown_output
    except Exception as e:
        return f"Error: {str(e)}"

#-------------------------
# GPT-4 Pipeline Functions
#-------------------------
client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def map_chunk_to_control(section_title, section_text, image_path, image_ocr):
    """
    For GPT-4 pipeline: send a prompt with the section and OCR text to GPT-4.
    Expect a JSON array output containing mapping objects (with the same keys).
    """
    system_message = f"""
    You are an expert in PCI-DSS compliance and network security. 
    Analyze the following section from a PCI-DSS ROC compliance document and the analyze the given image evidence.
    The document section (including its heading) contains requirement controls, guidelines, and auditor instructions. The image evidence
    has been provided by the audited organization as part of PCI DSS audit to prove that they fulfil a particular control
    of PCI DSS framework. Our task is to find out corresponding to which control this evidence was provided"""
    prompt = f"""
    Your task:
    1. Read and understand the given section carefully.
    2. Determine all specific requirement/control(s) from the section that are image evidence corresponds to.
    3. For each one, provide the exact control/requirement code (e.g., "Requirement 8.2.1.a").
    4. Provide a short excerpt from the section that describes that requirement.
    5. Explain briefly why the image evidence satisfies this requirement.
    6. Note if any aspects of the requirement are not fully satisfied.
    Always return your answer as a JSON array of objects with exactly these keys:
    "control_code": string,
    "description": string,
    "explanation": string,
    "missing_aspects": string
    Output only the JSON array with no extra text.
    Do not change the output format.
    Do not change the JSON key names.
    Do not change the data type of JSON objects, always return a string.
    For "missing_aspects", always return a string "",  do not return null or list[].

    Section Title: {section_title}
    Section Content:
    {section_text}

    Image OCR:
    {image_ocr}
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=500,
            temperature=0.2
        )
        output = response.choices[0].message.content.strip()
        if not output:
            print(f"[{section_title}] GPT-4 mapping returned empty output.")
            return []
        json_match = re.search(r"\[.*\]", output, re.DOTALL)
        if not json_match:
            print(f"[{section_title}] GPT-4 mapping did not return a JSON array.")
            return []
        json_str = json_match.group(0)
        mapping_list = json.loads(json_str)
        if isinstance(mapping_list, list):
            return [m for m in mapping_list if m.get("control_code")]
        else:
            return []
    except Exception as e:
        print(f"[{section_title}] GPT-4 mapping error: {e}")
        return []

def run_gpt4_pipeline(image_path, ocr_text):
    """Handle GPT-4 pipeline processing"""
    try:
        sections = load_json_file(pdf_sections_file)
        
        all_mappings = {}
        for title, text in sections.items():
            response = map_chunk_to_control(title, text, image_path, ocr_text)
            try:
                mappings = json.loads(response)
                if isinstance(mappings, list):
                    all_mappings[title] = mappings
            except json.JSONDecodeError as e:
                print(f"Error parsing response: {e}")
        
        aggregated = aggregate_mappings(all_mappings)
        markdown_output = format_mappings_to_markdown(all_mappings)
        save_json_file(all_mappings, "gpt4_mappings.json")
        return markdown_output
    except Exception as e:
        return f"Error: {str(e)}"

#-------------------------
# Gradio Interface
#-------------------------
with gr.Blocks(title="PCI-DSS Compliance Analyzer") as app:
    gr.Markdown("# PCI-DSS Compliance Mapping Analyzer")
    gr.Markdown("Upload an image of your network/system to analyze PCI-DSS compliance")
    
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="filepath", label="Upload Network/System Image")
            ocr_status = gr.Textbox(label="OCR Status", interactive=False)
    
    with gr.Row():
        open_source_btn = gr.Button("Run Open-Source Pipeline", variant="primary")
        gpt4_btn = gr.Button("Run GPT-4 Pipeline", variant="secondary")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Open-Source Results")
            open_source_output = gr.Markdown(label="Mapped Requirements", show_label=False)
        
        with gr.Column():
            gr.Markdown("### GPT-4 Results")
            gpt4_output = gr.Markdown(label="Mapped Requirements", show_label=False)
    
    # Event handlers
    image_input.change(
        fn=process_image_and_extract_ocr,
        inputs=image_input,
        outputs=ocr_status,
        show_progress="full"
    )
    
    open_source_btn.click(
        fn=run_open_source_pipeline,
        inputs=[image_input, ocr_status],
        outputs=open_source_output,
        show_progress="full"
    )
    
    gpt4_btn.click(
        fn=run_gpt4_pipeline,
        inputs=[image_input, ocr_status],
        outputs=gpt4_output,
        show_progress="full"
    )

if __name__ == "__main__":
    app.launch()