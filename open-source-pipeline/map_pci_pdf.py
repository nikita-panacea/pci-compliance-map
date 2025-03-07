import os
import json
import re
import time
import ollama  # Make sure ollama is installed and configured

def load_json_file(filename):
    """Load JSON data from a file."""
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json_file(data, filename):
    """Save JSON data to a file."""
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# def clean_text(text):
#     """Replace special/unusual characters with standard equivalents."""
#     replacements = {
#         "–": "-",
#         "—": "-",
#         "’": "'",
#         "“": '"',
#         "”": '"',
#     }
#     for orig, repl in replacements.items():
#         text = text.replace(orig, repl)
#     # Normalize whitespace
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()

def map_section_to_requirement(section_title, section_text, image_analysis):
    """
    Sends a prompt to the reasoning model (deepseek-r1:7b via ollama)
    including the section text and the image analysis report.
    Returns a mapping as a dict with keys:
      - "control_code"
      - "description"
      - "explanation"
      - "missing_aspects"
    If no mapping is found, returns {}.
    """
    # Skip sections that are too small
    if len(section_text.strip()) < 10:
        return {}

    prompt = f"""
    You are an expert in PCI-DSS compliance and network security. Analyze the following section from a PCI-DSS ROC compliance document and the provided detailed image analysis report.
    The document section (including its heading) contains requirement controls, guidelines, and auditor instructions.
    Your task:
    1. Determine which specific requirement/control from the section is satisfied by the image evidence.
    2. Provide the exact control/requirement code (e.g., "Requirement 8.2.1.a").
    3. Provide a short excerpt from the section that describes that requirement.
    4. Explain briefly why the image evidence satisfies this requirement.
    5. Note if any aspects of the requirement are not fully satisfied.
    Return your answer as a JSON object with exactly these keys:
    "control_code": string,
    "description": string,
    "explanation": string,
    "missing_aspects": string

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
            return {}
        # Extract the JSON portion from the output
        json_match = re.search(r"\{.*\}", output, re.DOTALL)
        if not json_match:
            print(f"[{section_title}] No JSON found in output.")
            return {}
        json_str = json_match.group(0)
        mapping = json.loads(json_str)
        if mapping.get("control_code"):
            return mapping
        else:
            return {}
    except Exception as e:
        print(f"[{section_title}] Mapping error: {e}")
        return {}

def aggregate_mappings(mapping_list):
    """
    Aggregate mapping results from multiple sections into a single dictionary.
    For duplicate control codes, merge the explanations.
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
        else:
            aggregated[code] = {
                "description": mapping.get("description", ""),
                "explanation": mapping.get("explanation", ""),
                "missing_aspects": mapping.get("missing_aspects", "")
            }
    return aggregated

def main():
    # Filenames for input JSON files
    pdf_sections_file = "processed_pci_chunks.json"  # Processed PDF sections (dict: heading -> text)
    image_analysis_file = "vision_analysis.json"  # Image analysis report (e.g., {"analysis": "..."})
    output_file = "mapping_results.json"

    # Load PDF sections
    if not os.path.exists(pdf_sections_file):
        print("Error: PDF sections file not found. Process the PDF first.")
        return
    sections = load_json_file(pdf_sections_file)
    
    # Load image analysis report
    if not os.path.exists(image_analysis_file):
        print("Error: Image analysis file not found.")
        return
    image_analysis_data = load_json_file(image_analysis_file)
    # Assume the image analysis report is stored under the key "analysis"
    image_analysis = image_analysis_data.get("analysis", "")
    if not image_analysis:
        print("Error: Image analysis content is empty.")
        return

    # Process each section
    mapping_results = {}
    total_sections = len(sections)
    processed_sections = 0
    partial_results = {}
    for title, text in sections.items():
        processed_sections += 1
        # clean_section = clean_text(text)
        print(f"Processing section [{processed_sections}/{total_sections}]: {title}")
        mapping = map_section_to_requirement(title, text, image_analysis)
        if mapping:
            mapping_results[title] = mapping
            print(f"Mapped control for section '{title}': {mapping.get('control_code')}")
        else:
            print(f"No mapping found for section '{title}'.")
        # Save partial results after processing each section
        partial_results[title] = mapping
        save_json_file(partial_results, output_file)
        print(f"Saved partial mapping results to {output_file}")
        # Optionally, add a short delay to ease resource usage
        time.sleep(0.5)
    
    aggregated = aggregate_mappings(list(mapping_results.values()))
    print("Final Aggregated Mappings:")
    print(json.dumps(aggregated, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
