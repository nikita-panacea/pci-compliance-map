import os
import json
import re
import ollama  # Ensure ollama is installed and configured

def load_json(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def map_requirement_for_chunk(pdf_chunk, image_analysis_report):
    """
    Use an open-source LLM (via ollama, e.g. the phi3:mini model) to analyze a PDF chunk
    along with the image analysis report and determine if the evidence satisfies a PCI-DSS requirement.
    The prompt instructs the model to return a JSON object with keys:
       "control_code": e.g., "Requirement 8.2.1.a"
       "description": an excerpt from the PDF that describes the requirement
       "explanation": a brief explanation of why the image evidence satisfies this requirement.
    If no mapping is found, return {}.
    """
    prompt = f"""
    You are a PCI-DSS compliance and network security expert.
Analyze the following section from the PCI-DSS ROC Compliance Document and the detailed image analysis report provided.
The document section contains instructions, guidelines, and requirement controls.
The image analysis report contains the observed evidence.
Your task is to determine if the evidence in the image satisfies any specific control requirement mentioned in the section.
If yes, return a JSON object with exactly the following keys (and no additional text):
  "control_code": a string representing the requirement/control code (for example, "Requirement 8.2.1.a"),
  "description": a short excerpt from the document section that describes the requirement,
  "explanation": a brief explanation of why the image evidence satisfies this requirement.
If no requirement is satisfied in this section, return an empty dicctionary.

Document Section:
------------------
{pdf_chunk}

Image Analysis Report:
------------------------
{image_analysis_report}
"""
    try:
        response = ollama.generate(
            model="deepseek-r1:7b",#"phi3:mini",
            prompt=prompt,
            options={'temperature': 0.1}
        )
        output = response.get("response", "").strip()
        if not output:
            print("Empty mapping output for chunk.")
            return {}
        # Use regex to extract the JSON object from the output.
        json_match = re.search(r"\{.*\}", output, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            print("No JSON found in output:", output)
            return {}
        mapping = json.loads(json_str)
        return mapping if mapping.get("control_code") else {}
    except Exception as e:
        print(f"Mapping error for chunk: {e}")
        return {}

def aggregate_mappings(mappings):
    """
    Aggregate mapping dictionaries (from each relevant chunk) into a deduplicated dictionary keyed by control code.
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

def main():
    # Filenames for processed PDF chunks and image analysis report
    pdf_chunks_file = "pci_chunks.json"         # JSON file with processed PDF chunks (sections/sub-sections)
    image_analysis_file = "vision_analysis.json"   # JSON file with detailed image analysis report

    if not os.path.exists(pdf_chunks_file):
        print("Error: Processed PDF chunks file not found. Run your PDF processing pipeline first.")
        return
    if not os.path.exists(image_analysis_file):
        print("Error: Image analysis report file not found. Ensure it is saved as JSON.")
        return

    processed_sections = load_json(pdf_chunks_file)
    image_analysis_data = load_json(image_analysis_file)
    # Convert image analysis to string (if not already)
    if isinstance(image_analysis_data, dict):
        image_analysis_report = json.dumps(image_analysis_data, indent=2)
    else:
        image_analysis_report = str(image_analysis_data)

    mapping_results = []
    for section_title, chunks in processed_sections.items():
        print(f"Processing section: {section_title}")
        for idx, chunk in enumerate(chunks, start=1):
            print(f"Mapping chunk {idx} in section '{section_title}'...")
            mapping = map_requirement_for_chunk(chunk, image_analysis_report)
            if mapping and mapping.get("control_code"):
                mapping_results.append(mapping)

    if not mapping_results:
        print("No mapped controls found for the given image evidence.")
        return

    aggregated = aggregate_mappings(mapping_results)
    print("Aggregated Mapped Controls:")
    for code, details in aggregated.items():
        print(f"Control: {code}")
        print(f"Description: {details.get('description')}")
        print(f"Explanation: {details.get('explanation')}\n")

if __name__ == "__main__":
    main()
