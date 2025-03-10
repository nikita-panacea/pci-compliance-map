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

# Load PDF sections
pdf_sections_file = "open-source-pipeline\processed_pci_chunks.json"


#------------------
# Image OCR 
#------------------
class OCRProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
    
    def process_image(self, image_path, output_file="ocr_results.json"):
        # image = Image.open(image_path).convert("RGB")
        # if max(image.size) > 1024:
        #     image.thumbnail((1024, 1024))
        print("Extracting text from image...")
        result = self.reader.readtext(image_path, detail=0)
        print("Text extraction complete.")
        with open(output_file, "w") as f:
            json.dump({"text": "\n".join(result)}, f)
        return output_file

#------------------------
# Open Source Pipeline
#------------------------

#------------------
# Image Analysis
#------------------
# Visual understanding with Minicpm/Llama3.2-Vision

class VisualAnalyzer:
    def __init__(self):
        self.model_name = "minicpm-v:latest" # llama3.2-vision:latest
        
    def analyze(self, image_path, image_ocr, output_file="vision_analysis_minicpm.json"):
        prompt = f"""
        You are an expert in PCI-DSS compliance network. security. 
        Provide a detailed report on the given image.
        Describe this image in detail, using at least a 1000 words or more, for compliance analysis:
        - List all the systems and devices present and their roles in the image.
        - List all connections present explaining network architecture.
        - Explain the network topology and any security measures in place.
        - Identify any potential vulnerabilities or security risks.
        Context text:
        {image_ocr}
        """
        print("Analyzing image generating report...")
        response = ollama.generate(
            model=self.model_name,
            prompt=prompt,
            images=[image_path]
        )
        print("Analysis complete.")
        with open(output_file, "w") as f:
            json.dump({"analysis": response['response']}, f)
        return output_file

# Initialize components
processor = OCRProcessor()
analyzer = VisualAnalyzer()

#-------------------------
# Compliance Mapping
#-------------------------
def map_section_to_requirement(section_title, section_text, image_analysis):
    """
    Sends a prompt to the reasoning model (deepseek-r1:7b via ollama)
    including the section text and the image analysis report.
    Now, it asks the model to return a JSON array containing mapping objects.
    Each mapping object should have the keys:
      - "control_code": string (e.g., "Requirement 8.2.1.a")
      - "description": a short excerpt from the section that describes the requirement
      - "explanation": a brief explanation of why the image evidence satisfies the requirement
      - "missing_aspects": a note on any aspects that are not fully satisfied
    If no mappings are found, the model should return an empty array: [].
    """
    # Skip sections that are too small
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
        # Use regex to extract a JSON array from the output.
        json_match = re.search(r"\[.*\]", output, re.DOTALL)
        if not json_match:
            print(f"[{section_title}] No JSON array found in output.")
            return []
        json_str = json_match.group(0)
        mapping_list = json.loads(json_str)
        if isinstance(mapping_list, list):
            # Filter out any mappings that don't have a control_code.
            return [m for m in mapping_list if m.get("control_code")]
        else:
            return []
    except Exception as e:
        print(f"[{section_title}] Mapping error: {e}")
        return []

def aggregate_mappings(mapping_lists):
    """
    Aggregate mapping objects from multiple sections into a deduplicated dictionary keyed by control code.
    For duplicate control codes, merge explanations and missing_aspects.
    """
    aggregated = {}
    for mapping in mapping_lists:
        code = mapping.get("control_code")
        if not code:
            continue
        if code in aggregated:
            # Merge explanations and missing aspects if not already present.
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
        # Perform image analysis
        analysis_report_path = analyzer.analyze(image_path)
        with open(analysis_report_path, 'r') as f:
            analysis_data = json.load(f)
        image_analysis = analysis_data['analysis']
        
        # Load PDF sections
        if not os.path.exists(pdf_sections_file):
            print("Error: PDF sections file not found. Process the PDF first.")
        sections = load_json_file(pdf_sections_file)
        
        all_mappings = []
        total_sections = len(sections)
        processed_sections = 0
        partial_results = {}
        output_file = "open_source_mappings.json"
        for title, text in sections.items():
            processed_sections += 1
            #clean_section = clean_text(text)
            print(f"Processing section [{processed_sections}/{total_sections}]: {title}")
            mappings = map_section_to_requirement(title, text, image_analysis)
            if mappings:
                all_mappings.extend(mappings)
                print(f"Mapped {len(mappings)} control(s) for section '{title}'.")
            else:
                print(f"No mapping found for section '{title}'.")
            partial_results[title] = mappings
            save_json_file(partial_results, output_file)
            print(f"Saved partial mapping results to {output_file}")
            time.sleep(0.5)
        
        aggregated = aggregate_mappings(all_mappings)
        print("Final Aggregated Mappings:")
        print(json.dumps(aggregated, indent=2, ensure_ascii=False))
    except Exception as e:
        raise gr.Error(f"Open-source pipeline error: {str(e)}")



#-----------------------
# GPT-4 Pipeline
#-----------------------
client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def image_to_data_url(image_path: str) -> str:
    """
    Convert an image to a data URL.
    """
    with open(image_path, 'rb') as f:
        image_data = f.read()
    return f"data:image/png;base64,{base64.b64encode(image_data).decode()}"

def map_chunk_to_control(section_title, section_text, image_path: str, image_ocr: str) -> str:
    """
    Use the aggregated context (all relevant sections), the OCR text, and the image
    to have GPT-4 Vision map the screenshot to specific PCI-DSS rule(s).
    """
    system_message = f"""
    You are an expert in PCI-DSS compliance and network security. 
    You are given a section extracted from the PCI-DSS Report on Compliance Template containing the controls and requirements.
    A client has provided a screenshot showing details of their network and security configuration.
    Analyze the image and identify which specific control requirement is being addressed.
    Please be specific in your mapping.
    """
    prompt = f"""
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
    Content OCR Text:
    {image_ocr}

    """
    image_data_url = image_to_data_url(image_path)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}}
        ]}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",  
        messages=messages,
        max_tokens=500
    )
    return response.choices[0].message.content

def run_gpt4_pipeline(image_path, ocr_text):
    """Handle GPT-4 pipeline processing"""
    try:
        # Load PDF sections
        if not os.path.exists(pdf_sections_file):
            print("Error: PDF sections file not found. Process the PDF first.")
        sections = load_json_file(pdf_sections_file)
        
        all_mappings = []
        total_sections = len(sections)
        processed_sections = 0
        partial_results = {}
        output_file = "gpt4_mappings.json"
        for title, text in sections.items():
            processed_sections += 1
            #clean_section = clean_text(text)
            print(f"Processing section [{processed_sections}/{total_sections}]: {title}")
        for title, text in sections.items():
            response = map_chunk_to_control(title, text, image_path, ocr_text)
            if response:
                all_mappings.extend(response)
                print(f"Mapped {len(response)} control(s) for section '{title}'.")
            else:
                print(f"No mapping found for section '{title}'.")
            partial_results[title] = response
            save_json_file(partial_results, output_file)
            print(f"Saved partial mapping results to {output_file}")
            time.sleep(0.5)
        
        aggregated = aggregate_mappings(all_mappings)
        print("Final Aggregated Mappings:")
        print(json.dumps(aggregated, indent=2, ensure_ascii=False))
    except Exception as e:
        raise gr.Error(f"GPT-4 pipeline error: {str(e)}")

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
            open_source_output = gr.JSON(label="Mapped Requirements", show_label=False)
        
        with gr.Column():
            gr.Markdown("### GPT-4 Results")
            gpt4_output = gr.JSON(label="Mapped Requirements", show_label=False)
    
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