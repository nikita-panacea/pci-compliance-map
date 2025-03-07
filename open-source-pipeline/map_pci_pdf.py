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
#     """Replace special/unusual characters with standard equivalents and normalize whitespace."""
#     replacements = {
#         "–": "-",
#         "—": "-",
#         "’": "'",
#         "“": '"',
#         "”": '"',
#     }
#     for orig, repl in replacements.items():
#         text = text.replace(orig, repl)
#     text = re.sub(r"\s+", " ", text)
#     return text.strip()

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

def main():
    # Filenames for input JSON files
    pdf_sections_file = "processed_pci_chunks.json"  # Processed PDF sections (dict: heading -> text)
    image_analysis_file = "vision_analysis_minicpm.json"  # Image analysis report (e.g., {"analysis": "..."})
    output_file = "mapping_results_lists.json"

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
    image_analysis = image_analysis_data.get("analysis", "")
    if not image_analysis:
        print("Error: Image analysis content is empty.")
        return

    all_mappings = []
    total_sections = len(sections)
    processed_sections = 0
    partial_results = {}
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

if __name__ == "__main__":
    main()
