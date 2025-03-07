# # pdf_processor.py
# import os
# import json
# import re
# import fitz  # PyMuPDF

# def process_pci_pdf(pdf_path, output_json="pci_processed_sections.json"):
#     """
#     Process the PCI-DSS ROC compliance PDF by:
#       1. Extracting the full text from the PDF.
#       2. Splitting the text into sections based on table-of-contents indicators.
#          We look for lines that start with either "Part ..." (for parts)
#          or section numbering like "1.1", "3.2.4", etc.
#       3. Saving the resulting sections into a JSON file.
      
#     Returns a dictionary mapping section headers to their text.
#     """
#     # Open PDF and extract all text
#     doc = fitz.open(pdf_path)
#     full_text = ""
#     for page in doc:
#         # "text" mode preserves the flow of the document
#         full_text += page.get_text("text") + "\n"
    
#     # Define a regex pattern to detect section headers.
#     # This pattern looks for lines that start with either "Part <Roman numeral>"
#     # or a numbering like "1.", "1.1", "3.2.4", etc.
#     pattern = re.compile(
#         r'(?=^(?:Part\s+[IVXLCDM]+|(?:\d+\.)+\d*\s))',
#         flags=re.MULTILINE
#     )
    
#     # Split the full text into sections based on the pattern.
#     sections = re.split(pattern, full_text)
#     # Remove any empty sections and strip whitespace.
#     sections = [sec.strip() for sec in sections if sec.strip()]
    
#     # Build a dictionary mapping each section header (first line) to its full text.
#     section_dict = {}
#     for sec in sections:
#         lines = sec.splitlines()
#         # Assume the first non-empty line is the header.
#         header = lines[0].strip()
#         section_dict[header] = sec
    
#     # Save the processed sections to a JSON file.
#     with open(output_json, "w", encoding="utf-8") as f:
#         json.dump(section_dict, f, ensure_ascii=False, indent=2)
    
#     return section_dict

# if __name__ == "__main__":
#     import sys
#     if len(sys.argv) < 2:
#         print("Usage: python pdf_processor.py path_to_pdf")
#     else:
#         pdf_path = sys.argv[1]
#         sections = process_pci_pdf(pdf_path)
#         print(f"Processed {len(sections)} sections and saved to 'pci_processed_sections.json'.")

#-----------------------------------------------------------------------------#
# process_pci_pdf.py
import fitz  # PyMuPDF
import re
import json
import sys

def extract_text_from_pdf(pdf_path):
    """Extract all text from the PDF using PyMuPDF."""
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    return full_text

def split_text_by_headings(text):
    """
    Split the text into sections based on headings. This function uses a regex pattern
    that is designed to capture common heading formats in the PCI-DSS ROC document,
    such as lines starting with "Part" (followed by roman numerals) or numerical headings
    (e.g., "1", "1.1", "Requirement 1: ...", etc.).
    
    Returns a dict mapping each heading to its corresponding text.
    """
    # Define a regex pattern to match headings.
    # This pattern matches lines that:
    # - Start with "Part" followed by a roman numeral and some text, or
    # - Start with a digit (and optional dot/sublevel) followed by text.
    pattern = re.compile(r"^(Part\s+[IVXLCDM]+\s+.*|(?:\d+(?:\.\d+)*\s+.*?Requirement.*|(?:\d+(?:\.\d+)*\s+.*))$)", re.MULTILINE)
    
    # Find all headings:
    headings = pattern.findall(text)
    if not headings:
        print("No headings found; returning full text as one section.")
        return {"Full Document": text}
    
    # Use re.split to break text into segments. The capturing group makes sure headings are kept.
    segments = re.split(pattern, text)
    sections = {}
    current_heading = "Introduction"  # default if text appears before the first heading
    # re.split returns: [pre-heading text, heading1, text1, heading2, text2, ...]
    for i, seg in enumerate(segments):
        seg = seg.strip()
        if i % 2 == 1:
            # This segment is a heading.
            current_heading = seg
            sections[current_heading] = ""
        else:
            # This is the text following the last heading (or before any heading)
            if current_heading in sections:
                sections[current_heading] += seg + "\n"
            else:
                sections[current_heading] = seg + "\n"
    return sections

def main(pdf_path, output_json):
    print("Extracting text from PDF...")
    full_text = extract_text_from_pdf(pdf_path)
    print("Splitting text into sections using headings...")
    sections = split_text_by_headings(full_text)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)
    print(f"Processed PDF and saved sections to {output_json}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python process_pci_pdf.py <path_to_pdf> <output_json>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    output_json = sys.argv[2]
    main(pdf_path, output_json)
