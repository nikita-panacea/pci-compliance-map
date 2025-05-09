from PIL import Image
import easyocr
import json
import ollama
import os
import re
import time
import base64
from mimetypes import guess_type
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr
#import soundfile 

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
pdf_sections_file = "test_code/req_sections.json" #"test_code/pdf_split_sect.json" 

#------------------
# Image OCR 
#------------------
class OCRProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True) #NOTE: True on Omen, False on local cpu
    
    def process_image(self, image_path, ocr_file="ocr_results.json"):
        print("Extracting text from image...")
        result = self.reader.readtext(image_path, detail=0)
        print("Text extraction complete.")
        with open(ocr_file, "w") as f:
            json.dump({"text": "\n".join(result)}, f)
        return ocr_file

#------------------------------------------------
# Open Source Pipeline (MiniCPM-V + DeepSeek-R1)
#-------------------------------------------------

# Image Analysis
class VisualAnalyzer:
    def __init__(self):
        self.model_name = "minicpm-v:latest"  # or "llama3.2-vision:latest"
        
    # - List all systems and devices present and their roles.
    # - Describe all network connections and architecture.
    # - Explain the network topology and security measures.
    # - Identify any potential vulnerabilities or security risks.    
    def analyze(self, image_path, image_ocr, img_analysis_file="vision_analysis_minicpm.json"):
        prompt = f"""
        You are an expert in PCI-DSS compliance network security. You are provided with image evidence. 
        The image evidence has been provided by the audited organization as part of PCI DSS audit to prove 
        that they fulfil a particular control of PCI DSS framework.
        Provide a detailed report on the given image for compliance analysis:
        - List and name all components present in the image.
        - Describe the information presented in the image in detail.
        - Explain the information on security measures from the image.
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
        with open(img_analysis_file, "w") as f:
            json.dump({"analysis": response['response']}, f)
        return img_analysis_file

# Initialize components
processor = OCRProcessor()
# analyzer = VisualAnalyzer()

#------------------------------
# Compliance Mapping Functions
#------------------------------

def sanitize_mapping(mapping):
    """Ensure all fields are strings and handle list values"""
    if 'missing_aspects' in mapping:
        if isinstance(mapping['missing_aspects'], list):
            # Convert list to comma-separated string
            mapping['missing_aspects'] = ', '.join(mapping['missing_aspects'])
        elif not isinstance(mapping['missing_aspects'], str):
            # Convert other types to string
            mapping['missing_aspects'] = str(mapping['missing_aspects'])
    # Ensure all fields exist and are strings
    return {
        'control_code': str(mapping.get('control_code', '')),
        'description': str(mapping.get('description', '')),
        'explanation': str(mapping.get('explanation', '')),
        'missing_aspects': str(mapping.get('missing_aspects', ''))
    }

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
    The document section (including its heading) contains requirement controls, guidelines, and auditor instructions. The image evidence
    has been provided by the audited organization as part of PCI DSS audit to prove that they fulfil a particular control
    of PCI DSS framework. Our task is to find out corresponding to which control this evidence was provided.
    Your task:
    1. Read and understand the given section carefully.
    2. Determine all specific requirement/control(s) from the section that the image evidence corresponds to.
    3. For each one, provide the exact control/requirement code from the document section.
    4. Provide a short excerpt from the section that describes that requirement.
    5. Explain briefly why the image evidence satisfies this requirement.
    6. Note if any aspects of the requirement are not fully satisfied by the image evidence.
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
        try:
            mapping_list = json.loads(json_str)
            if isinstance(mapping_list, list):
                # Sanitize each mapping
                valid_mappings = []
                for m in mapping_list:
                    try:
                        cleaned = sanitize_mapping(m)
                        if cleaned['control_code']:
                            valid_mappings.append(cleaned)
                    except Exception as e:
                        print(f"Invalid mapping format: {e}")
                return valid_mappings
            return []
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {e}")
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
            # Safely handle missing aspects
            missing = str(control.get('missing_aspects', '')).strip()
            
            markdown_output += f"**Control Code**: {control['control_code']}\n\n"
            markdown_output += f"**Description**: {control['description']}\n\n"
            markdown_output += f"**Explanation**: {control['explanation']}\n\n"
            if missing:
                markdown_output += f"**Missing Aspects**: {missing}\n\n"
            markdown_output += "---\n"
    return markdown_output

#-------------------------
# Processing Functions
#-------------------------
def process_image_and_extract_ocr(image_path, progress=gr.Progress()):
    progress(0, desc="Starting OCR processing...")
    try:
        ocr_file = processor.process_image(image_path)
        progress(0.3, desc="OCR completed")
        # Load ocr file:
        with open(ocr_file, "r") as f:
            image_ocr = json.load(f)["text"]
        return image_ocr
    except Exception as e:
        raise gr.Error(f"OCR Error: {str(e)}")

def run_open_source_pipeline(image_path, progress=gr.Progress()):
    try:
        # OCR Processing
        progress(0, desc="Extracting text from image...")
        #ocr_output = processor.process_image(image_path)
        ocr_text = process_image_and_extract_ocr(image_path)
        # print(ocr_text)
        
        # Image Analysis
        progress(0.3, desc="Analyzing image...")
        image_analysis_file = analyzer.analyze(image_path, ocr_text)
        image_analysis_data = load_json_file(image_analysis_file)
        image_analysis = image_analysis_data.get("analysis", "")
        # print(image_analysis)
        
        # Load PDF sections
        sections = load_json_file(pdf_sections_file)
        total_sections = len(sections)
        
        # Process sections with progress
        #all_mappings = {}
        all_mappings=[]
        partial_results = {}
        markdown_output = ""
        processed_sections = 0
        for idx, (title, text) in enumerate(sections.items()):
            progress(0.4 + (idx/total_sections)*0.6, 
                    desc=f"Mapping requirements ({idx+1}/{total_sections})...")
            processed_sections += 1
            print(f"Processing section [{processed_sections}/{total_sections}]: {title}")
            mappings = map_section_to_requirement(title, text, image_analysis)
            if mappings:
                # all_mappings[title] = mappings
                all_mappings.extend(mappings)
                print(f"Mapped {len(mappings)} control(s) for section '{title}'.")
            else:
                print(f"No mapping found for section '{title}'.")    
            # Update results incrementally
            partial_results[title] = mappings
            section_md = format_mappings_to_markdown({title: mappings})
            markdown_output += section_md + "\n\n"
            yield markdown_output  # Yield partial results
            time.sleep(0.1)  # For smooth UI updates
        
        aggregated = aggregate_mappings(all_mappings)
        save_json_file(aggregated, "open_source_mappings.json")
        
    except Exception as e:
        yield f"Error: {str(e)}"


##-------------------------
# Janus Pro 7B Pipeline
#-------------------------
from transformers import AutoModelForCausalLM
# from Janus.janus.models import MultiModalityCausalLM, VLChatProcessor
# from Janus.janus.utils.io import load_pil_images
import torch

class Janus7BProcessor:
    def __init__(self):
        self.model_path = "deepseek-ai/Janus-Pro-7B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model and processor
        self.vl_chat_processor = VLChatProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.vl_chat_processor.tokenizer
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).to(self.device).eval()

    def generate_response(self, prompt_text, image_path):
        # Create conversation format with image PATH instead of PIL Image
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n{prompt_text}",
                "images": [image_path],  # Pass path instead of Image object
            },
            {"role": "<|Assistant|>", "content": ""},
        ]
        
        # Process inputs using processor's image loading
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.vl_chat_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.tokenizer.eos_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=1024,
                do_sample=False,
                use_cache=True,
            )
        
        # Decode response
        full_response = self.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        return full_response.split("<|Assistant|>")[-1].strip()


# Initialize Janus processor
# janus_processor = Janus7BProcessor()

def map_chunk_to_control_janus(section_title, section_text, image_path): #, image_ocr
    """
    Janus Pro 7B version of control mapping
    """
    system_message = f"""
    You are an expert in PCI-DSS compliance and network security. 
    Analyze the following section from a PCI-DSS ROC compliance document and the analyze the given image evidence.
    The document section (including its heading) contains requirement controls, guidelines, and auditor instructions. The image evidence
    has been provided by the audited organization as part of PCI DSS audit to prove that they fulfil a particular control
    of PCI DSS framework. Our task is to find out corresponding to which control this evidence was provided"""
    prompt = f"""
    {system_message}
    
    Section Title: {section_title}
    
    Section Content: {section_text}

    Your task:
    1. Read and understand the given section carefully.
    2. Determine all specific requirement/control(s) from the section that are image evidence corresponds to.
    3. For each one, provide the exact control/requirement code from the document section.
    4. Provide a short excerpt from the section that describes that requirement.
    5. Explain briefly why the image evidence satisfies this requirement.
    6. Note if any aspects of the requirement are not fully satisfied.
    Always return your answer as a JSON array of objects with exactly these keys:
    "control_code": string,
    "description": string,
    "explanation": string,
    "missing_aspects": string
    Output only the JSON array with no extra text.
    """
    
    try:
        raw_response = janus_processor.generate_response(prompt, image_path)
        # print(raw_response)
        
        # Extract JSON array
        json_match = re.search(r"\[.*\]", raw_response, re.DOTALL)
        if not json_match:
            print(f"[{section_title}] No JSON array found")
            return []
            
        json_str = json_match.group(0)
        mapping_list = json.loads(json_str)
        
        # Validate and sanitize
        valid_mappings = []
        for m in mapping_list:
            try:
                cleaned = sanitize_mapping(m)
                if cleaned['control_code']:
                    valid_mappings.append(cleaned)
            except Exception as e:
                print(f"Invalid Janus mapping: {e}")
        
        return valid_mappings
        
    except Exception as e:
        print(f"[{section_title}] Janus error: {e}")
        return []

def run_janus_pipeline(image_path, progress=gr.Progress()):
    try:
        # OCR Processing
        progress(0, desc="Extracting text from image...")
        ocr_text = processor.process_image(image_path)
        
        # Load PDF sections
        sections = load_json_file(pdf_sections_file)
        total_sections = len(sections)
        
        all_mappings = {}
        markdown_output = ""
        for idx, (title, text) in enumerate(sections.items()):
            progress((idx+1)/total_sections, 
                    desc=f"Processing section {idx+1}/{total_sections}...")
            
            response = map_chunk_to_control_janus(title, text, image_path)#, ocr_text
            if response:
                all_mappings[title] = response
                # Update results incrementally
                section_md = format_mappings_to_markdown({title: response})
                markdown_output += section_md + "\n\n"
                yield markdown_output
                time.sleep(0.1)
        
        save_json_file(all_mappings, "janus_mappings.json")
        
    except Exception as e:
        yield f"Error: {str(e)}"
    #return "Janus Results"

#-----------------------------
# Phi 4 Multimodal Pipeline
#-----------------------------

import requests
import torch
from PIL import Image
#import soundfile as sf
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

class Phi4Processor:
    def __init__(self):
        self.model_path = "microsoft/Phi-4-multimodal-instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize model components
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True, 
            attn_implementation='flash_attention_2',
        ).cuda()
        
        # Configure generation parameters
        self.generation_config = GenerationConfig.from_pretrained(self.model_path)
        # self.generation_config.max_new_tokens = 1024
        # self.generation_config.temperature = 0.7
        # self.generation_config.top_p = 0.9

    def generate_response(self, prompt_text, image_path):
        """Process image and generate response with Phi-4"""
        try:
            # Load and convert image with explicit size validation
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
                
            image = Image.open(image_path)
            
            user_prompt = '<|user|>'
            assistant_prompt = '<|assistant|>'
            prompt_suffix = '<|end|>'
            
            prompt = f'{user_prompt}<|image_1|>{prompt_text}{prompt_suffix}{assistant_prompt}'
            # Process inputs with enhanced validation
            inputs = self.processor(text=prompt, images=image, return_tensors='pt').to('cuda:0')

            # Generate with explicit tensor passing
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=1000,
                generation_config=self.generation_config,
            )
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]

            # Decode and clean response
            response = self.processor.batch_decode(
                generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]
            
            return response
            
        except Exception as e:
            print(f"Phi-4 generation error: {e}")
            return ""

# Initialize Phi-4 processor
#phi4_processor = Phi4Processor()

def map_chunk_to_control_phi4(section_title, section_text, image_path, image_ocr):
    """
    Phi-4 version of compliance mapping
    """
    system_message = """You are an expert PCI-DSS compliance analyst. Analyze the following section from a PCI-DSS ROC compliance document and the provided detailed image analysis report.
    The document section (including its heading) contains requirement controls, guidelines, and auditor instructions.
    The image evidence has been provided by the audited organization as part of PCI DSS audit to prove that they fulfil a particular control
    of PCI DSS framework. Our task is to find out corresponding to which control this evidence was provided."""
    
    task_prompt = f"""
    {system_message}
    
    Section Title: {section_title}
    
    Section Content: {section_text}
    Context: {image_ocr}

    Your task:
    1. Read and understand the given section carefully.
    2. Determine all the specific requirement/control(s) from the section that are image evidence corresponds to. 
    3. Only return the control code(s) for the section if the image evidence provided satisfies the requirement/control.
    3. For each one, provide the exact control/requirement code from the document section, format Requirement x.x.x
    4. Provide a short excerpt from the document section that describes that requirement.
    5. Explain briefly why the image evidence satisfies this requirement.
    6. Note if any aspects of the requirement are not fully satisfied by the image evidence.
    3. Always return your answer as a JSON array of objects with exactly these keys and format:
       
        "control_code": string,
        "description": string,
        "explanation": string,
        "missing_aspects": string
        
    Output only the JSON array with no extra text.
      
    """
    
    try:
        # Generate and validate response
        raw_response = phi4_processor.generate_response(task_prompt, image_path)
        
        # Handle empty responses
        if not raw_response.strip():
            return []
            
        # Robust JSON extraction with error recovery
        json_str = re.search(r"\[\s*({.*?}\s*,?\s*)+]", raw_response, re.DOTALL)
        if json_str:
            try:
                return [
                    sanitize_mapping(m) 
                    for m in json.loads(json_str.group()) 
                    if isinstance(m, dict) and m.get('control_code')
                ]
            except json.JSONDecodeError:
                return []
        return []
        
    except Exception as e:
        print(f"[{section_title}] Phi-4 error: {e}")
        return []
            
    #     json_str = json_match.group(0)
    #     mapping_list = json.loads(json_str)
        
    #     # Validate and sanitize
    #     valid_mappings = []
    #     for m in mapping_list:
    #         try:
    #             cleaned = sanitize_mapping(m)
    #             if cleaned['control_code']:
    #                 valid_mappings.append(cleaned)
    #         except Exception as e:
    #             print(f"Invalid Phi-4 mapping: {e}")
        
    #     return valid_mappings
        
    # except Exception as e:
    #     print(f"[{section_title}] Phi-4 error: {e}")
    #     return []

def run_phi4_pipeline(image_path, progress=gr.Progress()):
    try:
        # OCR Processing
        progress(0, desc="Extracting text from image...")
        ocr_text = processor.process_image(image_path)
        
        # Load PDF sections
        sections = load_json_file(pdf_sections_file)
        total_sections = len(sections)
        
        all_mappings = {}
        markdown_output = ""
        for idx, (title, text) in enumerate(sections.items()):
            progress((idx+1)/total_sections, 
                    desc=f"Processing section {idx+1}/{total_sections}...")
            
            response = map_chunk_to_control_phi4(title, text, image_path, ocr_text)
            if response:
                all_mappings[title] = response
                # Update results incrementally
                section_md = format_mappings_to_markdown({title: response})
                markdown_output += section_md + "\n\n"
                yield markdown_output
                time.sleep(0.1)
        
        save_json_file(all_mappings, "phi4_mappings.json")
        
    except Exception as e:
        yield f"Error: {str(e)}"


#-------------------------
# GPT-4 Pipeline Functions
#-------------------------
client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def image_to_data_url(image_path: str) -> str:
    """Convert an image file to a base64-encoded data URL."""
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'
    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"

def map_gpt4_chunk_to_control(section_title, section_text, image_path, image_ocr):
    """
    For GPT-4 pipeline: send a prompt with the section and OCR text to GPT-4.
    Expect a JSON array output containing mapping objects (with the same keys).
    """
    system_message = f"""
    You are an expert in PCI-DSS compliance and network security. 
    Analyze the following section from a PCI-DSS ROC compliance document and the provided detailed image analysis report.
    The document section (including its heading) contains requirement controls, guidelines, and auditor instructions.
    The image evidence has been provided by the audited organization as part of PCI DSS audit to prove that they fulfil a particular control
    of PCI DSS framework. Our task is to find out corresponding to which control this evidence was provided.
    """
    prompt = f"""
    Your task:
    1. Read and understand the given section carefully.
    2. Determine all specific requirement/control(s) from the section that are image evidence corresponds to.
    3. For each one, provide the exact control/requirement code from the document section.
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
        try:
            mapping_list = json.loads(json_str)
            if isinstance(mapping_list, list):
                # Sanitize each mapping
                valid_mappings = []
                for m in mapping_list:
                    try:
                        cleaned = sanitize_mapping(m)
                        if cleaned['control_code']:
                            valid_mappings.append(cleaned)
                    except Exception as e:
                        print(f"Invalid GPT-4 mapping format: {e}")
                return valid_mappings
            return []
        except json.JSONDecodeError as e:
            print(f"GPT-4 JSON decode error: {e}")
            return []
    except Exception as e:
        print(f"[{section_title}] GPT-4 mapping error: {e}")
        return []

def run_gpt4_pipeline(image_path, progress=gr.Progress()):
    try:
        # OCR Processing
        progress(0, desc="Extracting text from image...")
        ocr_text = processor.process_image(image_path)
        
        # Load PDF sections
        sections = load_json_file(pdf_sections_file)
        total_sections = len(sections)
        
        all_mappings = {}
        markdown_output = ""
        for idx, (title, text) in enumerate(sections.items()):
            progress((idx+1)/total_sections, 
                    desc=f"Processing section {idx+1}/{total_sections}...")
            
            response = map_gpt4_chunk_to_control(title, text, image_path, ocr_text)
            if response:
                all_mappings[title] = response
                # Update results incrementally
                section_md = format_mappings_to_markdown({title: response})
                markdown_output += section_md + "\n\n"
                yield markdown_output  # Yield partial results
                time.sleep(0.1)
        
        save_json_file(all_mappings, "gpt4_mappings.json")
        
    except Exception as e:
        yield f"Error: {str(e)}"

#----------------------------------------
# Pixtral Multimodal Pipeline Functions
#-----------------------------------------
from mistralai import Mistral
mistral_api_key = 'WceN8y36a516WiLlxxjHcIUdKkEQhjFD'#os.getenv("MISTRAL_API_KEY")
mistral_model = "pixtral-12b-2409"
mistral_client = Mistral(api_key=mistral_api_key)

def encode_image_base64(image_path): 
	with open(image_path, "rb") as image_file:   
		return base64.b64encode(image_file.read()).decode("utf-8")

def map_pixtral_chunk_to_control(section_title, section_text, image_path, image_ocr):
    """
    For GPT-4 pipeline: send a prompt with the section and OCR text to GPT-4.
    Expect a JSON array output containing mapping objects (with the same keys).
    """
    system_message = f"""
    You are an expert in PCI-DSS compliance and network security. 
    Analyze the following section from a PCI-DSS ROC compliance document and the provided detailed image analysis report.
    The document section (including its heading) contains requirement controls, guidelines, and auditor instructions.
    The image evidence has been provided by the audited organization as part of PCI DSS audit to prove that they fulfil a particular control
    of PCI DSS framework. Our task is to find out corresponding to which control this evidence was provided.
    """
    prompt = f"""
    Your task:
    1. Read and understand the given section carefully.
    2. Determine all specific requirement/control(s) from the section that are image evidence corresponds to.
    3. For each one, provide the exact control/requirement code from the document section.
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
    image_data_url = encode_image_base64(image_path)
    messages = [
	{
		"role":"user",
		"content":[{"type": "text", "text": system_message}, {"type": "text", "text": prompt}, {"type": "image_url", "image_url": f"data:image/png;base64,{image_data_url}"}]
	}
]
    # messages = [
    #     {"role": "system", "content": system_message},
    #     {"role": "user", "content": [
    #         {"type": "text", "text": prompt},
    #         {"type": "image_url", "image_url": {"url": image_data_url}}
    #     ]}
    # ]
    try:
        pix_response = mistral_client.chat.complete(
            model=mistral_model,
            messages=messages,
            max_tokens=500,
            temperature=0.2
        )
        output = pix_response.choices[0].message.content.strip()
        if not output:
            print(f"[{section_title}] Pixtral mapping returned empty output.")
            return []
        json_match = re.search(r"\[.*\]", output, re.DOTALL)
        if not json_match:
            print(f"[{section_title}] Pixtral mapping did not return a JSON array.")
            return []
        json_str = json_match.group(0)
        try:
            mapping_list = json.loads(json_str)
            if isinstance(mapping_list, list):
                # Sanitize each mapping
                valid_mappings = []
                for m in mapping_list:
                    try:
                        cleaned = sanitize_mapping(m)
                        if cleaned['control_code']:
                            valid_mappings.append(cleaned)
                    except Exception as e:
                        print(f"Invalid Pixtral mapping format: {e}")
                return valid_mappings
            return []
        except json.JSONDecodeError as e:
            print(f"Pixtral JSON decode error: {e}")
            return []
    except Exception as e:
        print(f"[{section_title}] Pixtral mapping error: {e}")
        return []

def run_pixtral_pipeline(image_path, progress=gr.Progress()):
    try:
        # OCR Processing
        progress(0, desc="Extracting text from image...")
        ocr_text = processor.process_image(image_path)
        
        # Load PDF sections
        sections = load_json_file(pdf_sections_file)
        total_sections = len(sections)
        
        all_mappings = {}
        markdown_output = ""
        for idx, (title, text) in enumerate(sections.items()):
            progress((idx+1)/total_sections, 
                    desc=f"Processing section {idx+1}/{total_sections}...")
            
            response = map_pixtral_chunk_to_control(title, text, image_path, ocr_text)
            if response:
                all_mappings[title] = response
                # Update results incrementally
                section_md = format_mappings_to_markdown({title: response})
                markdown_output += section_md + "\n\n"
                yield markdown_output  # Yield partial results
                time.sleep(0.1)
        
        save_json_file(all_mappings, "pixtral_mappings.json")
        
    except Exception as e:
        yield f"Error: {str(e)}"



#-------------------------
# Gradio Interface
#-------------------------
with gr.Blocks(title="PCI-DSS Compliance Analyzer") as app:
    gr.Markdown("# PCI-DSS Compliance Mapping Analyzer")
    gr.Markdown("Upload an evidence image for compliance analysis")
    
    # Image Upload Section
    with gr.Row():
        image_input = gr.Image(type="filepath", label="Upload Image", height=300)
    
    # Pipeline Tabs
    with gr.Tabs() as tabs:
        # Open-Source Pipeline Tab
        with gr.Tab("MiniCPM-V + DeepSeek-R1"):
            with gr.Row():
                oss_btn = gr.Button("Run Open-Source Pipeline", variant="primary")
            with gr.Row():
                oss_output = gr.Markdown("### Open-Source Results will appear here")
        
        # GPT-4 Pipeline Tab
        with gr.Tab("GPT-4"):
            with gr.Row():
                gpt4_btn = gr.Button("Run GPT-4 Pipeline", variant="secondary")
            with gr.Row():
                gpt4_output = gr.Markdown("### GPT-4 Results will appear here")
        
        # Janus Pro Pipeline Tab
        with gr.Tab("Janus Pro 7B"):
            with gr.Row():
                janus_btn = gr.Button("Run Janus Pro Pipeline", variant="primary")
            with gr.Row():
                janus_output = gr.Markdown("### Janus Pro Results will appear here")
        
        # Phi-4 Pipeline Tab
        with gr.Tab("Phi-4 Multimodal"):
            with gr.Row():
                phi4_btn = gr.Button("Run Phi-4 Pipeline", variant="secondary")
            with gr.Row():
                phi4_output = gr.Markdown("### Phi-4 Results will appear here")
        
        # Pixtral Pipeline Tab
        with gr.Tab("Pixtral Multimodal"):
            with gr.Row():
                pix_btn = gr.Button("Run Pixtral Pipeline", variant="secondary")
            with gr.Row():
                pix_output = gr.Markdown("### Pixtral Results will appear here")

    # Event Handlers
    oss_btn.click(
        fn=run_open_source_pipeline,
        inputs=image_input,
        outputs=oss_output,
        show_progress=True
    )
    
    gpt4_btn.click(
        fn=run_gpt4_pipeline,
        inputs=image_input,
        outputs=gpt4_output,
        show_progress=True
    )
    
    janus_btn.click(
        fn=run_janus_pipeline,
        inputs=image_input,
        outputs=janus_output,
        show_progress=True
    )
    
    phi4_btn.click(
        fn=run_phi4_pipeline,
        inputs=image_input,
        outputs=phi4_output,
        show_progress=True
    )

    pix_btn.click(
        fn=run_pixtral_pipeline,
        inputs=image_input,
        outputs=pix_output,
        show_progress=True
    )

if __name__ == "__main__":
    app.launch()