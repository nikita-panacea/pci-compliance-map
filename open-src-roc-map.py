import fitz  # PyMuPDF
import chromadb
import re
import easyocr
from PIL import Image
import gradio as gr
import ollama
from transformers import DonutProcessor, VisionEncoderDecoderModel
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from collections import defaultdict
import time
from functools import lru_cache
os.environ["OMP_NUM_THREADS"] = "4"  # Add this line
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Add this too


# Configuration
DB_PATH = "pci_dss_db"
COLLECTION_NAME = "requirements"
EXTRACTION_MODEL = "naver-clova-ix/donut-base-finetuned-docvqa"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cache the embedding model
@lru_cache(maxsize=1)
def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

# In ComplianceMapper.map_requirements:
model = get_embedding_model()  # Instead of creating new instances

# 1. PDF Processing (Complete Preservation)
def process_and_store_pdf(pdf_path):
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    
    try:
        collection = chroma_client.get_collection(COLLECTION_NAME)
        return "Document already processed!"
    except:
        collection = chroma_client.create_collection(COLLECTION_NAME)

    # Extract text with layout preservation
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        page_text = page.get_text("text")
        if not page_text.strip():
            page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_IMAGES)
        full_text += page_text + "\n"

    # Split requirements with improved regex
    requirement_pattern = re.compile(
        r'(?:^|\n)(\d+\.\d+(?:\.\d+[a-z]?)?)\s*(.*?)\n(.*?)(?=\n\s*\d+\.\d|\Z)',
        flags=re.DOTALL
    )
    
    matches = requirement_pattern.findall(full_text)
    
    # Validate matches before processing
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

    # Validate requirements before embedding
    if not requirements:
        return "Error: No valid requirements extracted"
        
    # After requirement extraction in process_and_store_pdf
    print(f"Sample requirements extracted:")
    for req in requirements[:3]:
        print(f"{req['original_code']}: {req['title']}")
        print(req['description'][:100] + "...\n")

    # Generate embeddings with validation
    try:
        model = SentenceTransformer(EMBEDDING_MODEL)
        embeddings = model.encode(
            [req["full_text"] for req in requirements],
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        if len(embeddings) == 0 or embeddings.shape[0] != len(requirements):
            return f"Error: Embedding generation failed ({len(embeddings)} vs {len(requirements)})"
            
        collection.add(
            ids=[req["unique_code"] for req in requirements],
            embeddings=embeddings.tolist(),  # Direct numpy conversion
            documents=[req["full_text"] for req in requirements],
            metadatas=[{
                "original_code": req["original_code"],
                "title": req["title"]
            } for req in requirements]
        )
        
    except Exception as e:
        return f"Error during embedding: {str(e)}"

    try:
        verify_count = collection.count()
        if verify_count != len(requirements):
            return f"Error: Storage failed ({verify_count}/{len(requirements)} saved)"
    except Exception as e:
        return f"Verification failed: {str(e)}"

    return f"Processed {len(requirements)} requirements ({len(code_counter)} unique codes)"

# 2. Image Processing (Multimodal Extraction)
class ImageAnalyzer:
    def __init__(self):
        # Preload models only once
        self.processor = DonutProcessor.from_pretrained(EXTRACTION_MODEL)
        self.doc_model = VisionEncoderDecoderModel.from_pretrained(EXTRACTION_MODEL)
        self.doc_model.eval()  # Disable training mode
        self.general_ocr = easyocr.Reader(['en'], gpu=False)  # Force CPU
        
        # Safe ChromaDB initialization
        self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
        try:
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
        except:
            self.collection = None

    def analyze_image(self, image_path):
        if not self.collection:
            raise ValueError("Process PCI document first!")
        
        # Resize large images to max 1024px
        image = Image.open(image_path).convert("RGB")
        if max(image.size) > 1024:
            image.thumbnail((1024, 1024))

        # Document-style OCR with Donut
        image = Image.open(image_path).convert("RGB")
        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        task_prompt = "<s_docvqa><s_question>{What technical requirements are shown here?}</s_question><s_answer>"
        decoder_input_ids = self.processor.tokenizer(
            task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
        
        output = self.doc_model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=512,
        )
        # Corrected decoding line
        doc_text = self.processor.batch_decode(output)[0] # Directly decode the output tensor
        # Enhanced text cleaning
        doc_text = re.sub(r"<.*?>", "", doc_text)  # Remove any HTML tags
        # General OCR for other text
        try:
            general_text = self.general_ocr.readtext(image_path, detail=0)
        except Exception as e:
            general_text = [f"OCR Error: {str(e)}"]
        combined_text = f"Document Analysis: {doc_text}\nGeneral Text: {' '.join(general_text)}"
        
        # Visual understanding with LLaVA
        response = ollama.generate(
            model="llava:7b-v1.6-mistral-q4_K_M",
            prompt="Describe this technical image in detail for compliance analysis:",
            images=[image_path]
        )
        visual_description = response['response']
        
        return f"""Image Content Analysis:
        - Document Understanding: {doc_text}
        - OCR Extracted Text: {', '.join(general_text)}
        - Combinedd Text: {combined_text}
        - Visual Description: {visual_description}"""

# 3. Requirement Mapping Engine
class ComplianceMapper:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = None
        self.metadata_map = {}
        self._initialize_collection()

    def _initialize_collection(self):
        """Safer collection initialization with retries"""
        try:
            self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
            if self.collection.count() == 0:
                raise ValueError("Collection is empty")
                
            results = self.collection.get(include=["metadatas", "documents"])
            self.metadata_map = {
                id: (metadata, doc) for id, metadata, doc in zip(
                    results["ids"], 
                    results["metadatas"], 
                    results["documents"]
                )
            }
            print(f"Loaded {self.collection.count()} requirements from database")  # Debug
        except Exception as e:
            print(f"Collection initialization failed: {e}")
            self.collection = None
            self.metadata_map = {}

    def map_requirements(self, image_analysis):
        if not self.collection:
            return {"error": "PCI document not processed. Complete Step 1 first."}
        
        try:
            # Enhanced query with hybrid search
            model = get_embedding_model()
            print("Analyzing image content:", image_analysis[:200] + "...")  # Debug
            
            query_embedding = model.encode(
                image_analysis, 
                show_progress_bar=False,
                convert_to_tensor=True
            ).cpu().numpy().tolist()
            
            # Increase results and lower confidence threshold
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=15,  # Increased from 10
                include=["metadatas", "documents", "distances"],
                where={"original_code": {"$ne": ""}}
            )
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}
        
        # Add validation for query results
        if not results.get("metadatas") or not results.get("documents"):
            return {"error": "No matching requirements found in document"}
            
        print(f"Found {len(results['metadatas'][0])} potential matches")  # Debug
        
        # Enhanced grouping with similarity scores
        from collections import defaultdict
        grouped_results = defaultdict(list)
        for metadata, doc, score in zip(
            results["metadatas"][0], 
            results["documents"][0],
            results["distances"][0]
        ):
            original_code = metadata.get("original_code", "")
            if not original_code:
                continue
                
            similarity = 1 - score
            grouped_results[original_code].append({
                "text": doc,
                "title": metadata.get("title", "No title"),
                "score": similarity
            })

        # Process requirements with adjusted confidence threshold
        final_results = []
        for code, sections in grouped_results.items():
            avg_score = sum(s["score"] for s in sections) / len(sections)
            print(f"Processing {code} (avg score: {avg_score:.2f})")  # Debug
            
            # # Lower confidence threshold from 0.35 to 0.25
            # if avg_score < 0.25:
            #     print(f"Skipping {code} - low confidence")
            #     continue

            # Combine sections with context-aware selection
            combined_text = "\n".join([
                s["text"] for s in sorted(
                    sections,
                    key=lambda x: x["score"],
                    reverse=True
                )[:3]  # Take top 3 relevant sections
            ])[:3000]  # Truncate to context limit

            # Enhanced LLM prompt with examples
            prompt = f"""Analyze PCI-DSS compliance for network infrastructure.
            
            Requirement {code}:
            {combined_text}

            Observed Configuration:
            {image_analysis}

            Does the image demonstrate compliance with requirement {code}?
            Consider these indicators:
            - Security controls implementation
            - Network segmentation
            - Encryption mechanisms
            - Access restrictions

            Answer STRICTLY in this format:
            - [COMPLIANT/NON-COMPLIANT]: 
            - Reason: [Technical explanation]"""

            try:
                response = ollama.generate(
                    model="phi3:mini",#'deepseek-r1:7b',#
                    prompt=prompt,
                    options={'temperature': 0.1}
                )
                response_text = response['response'].strip()
                print(f"LLM response for {code}: {response_text}")  # Debug
                
                if "COMPLIANT" in response_text:
                    reasoning = response_text.split("- Reason:", 1)[-1].strip()
                    final_results.append({
                        "code": code,
                        "title": sections[0]["title"],
                        "description": combined_text[:500],  # Truncate for display
                        "explanation": reasoning,
                        "confidence": f"{avg_score:.0%}"
                    })
            except Exception as e:
                print(f"Error processing {code}: {str(e)}")

        # Return sorted results or empty array
        return sorted(final_results, key=lambda x: x["confidence"], reverse=True) if final_results else []
    
# 4. Fixed Gradio Interface with State Persistence
def create_interface():
    with gr.Blocks(title="PCI-DSS Compliance Mapper") as app:
        # Shared state components
        analyzer_state = gr.State(None)
        mapper_state = gr.State(None)
        processed_state = gr.State(False)  # Track document processing
        
        with gr.Tab("Document Setup"):
            gr.Markdown("## Step 1: Process PCI-DSS Document")
            pdf_input = gr.File(label="Upload ROC PDF")
            process_btn = gr.Button("Process Document")
            setup_status = gr.Textbox(label="Processing Status")
            
            def process_and_update(pdf_path):
                try:
                    result = process_and_store_pdf(pdf_path)
                    # Initialize new analyzer/mapper instances
                    new_analyzer = ImageAnalyzer()
                    new_mapper = ComplianceMapper()
                    return [
                        result,        # For setup_status
                        new_analyzer,  # For analyzer_state
                        new_mapper,    # For mapper_state
                        True           # For processed_state
                    ]
                except Exception as e:
                    return [f"Error: {str(e)}", None, None, False]
            
            process_btn.click(
                process_and_update,
                inputs=pdf_input,
                outputs=[setup_status, analyzer_state, mapper_state, processed_state]
            )

        with gr.Tab("Image Analysis"):
            gr.Markdown("## Step 2: Analyze Image")
            image_input = gr.Image(type="filepath", label="Upload Image")
            analyze_btn = gr.Button("Analyze Compliance")
            results = gr.JSON(label="Mapped Requirements")
            status = gr.Textbox(label="Processing Status")
            
            def analyze_wrapper(image_path, analyzer, mapper, is_processed):
                if not is_processed:
                    return {"error": "Process document first"}, "Not processed"
                if analyzer is None or mapper is None:
                    return {"error": "Components not initialized"}, "Failed"
                try:
                    start = time.time()
                    
                    # Add image validation
                    if not os.path.exists(image_path):
                        return {"error": "Image file not found"}, "Failed"
                        
                    analysis = analyzer.analyze_image(image_path)
                    mid = time.time()
                    
                    # Add analysis validation
                    if not analysis or len(analysis) < 50:
                        return {"error": "Image analysis failed"}, "Invalid analysis"
                        
                    output = mapper.map_requirements(analysis)
                    end = time.time()
                    
                    return (
                        output,
                        f"Processed in {end-start:.1f}s (OCR: {mid-start:.1f}s, Mapping: {end-mid:.1f}s)"
                    )
                except Exception as e:
                    return {"error": str(e)}, "Failed"
            
            analyze_btn.click(
                analyze_wrapper,
                inputs=[image_input, analyzer_state, mapper_state, processed_state],
                outputs=[results, status]
            )

    return app

# Execution with error handling
if __name__ == "__main__":
    # Set environment variables first
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    
    # Launch application
    try:
        create_interface().launch()
    except Exception as e:
        print(f"Error: {e}")
        print("Ensure you've processed the document first through the setup tab!")