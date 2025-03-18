import fitz  # PyMuPDF
import re
import json
import sys

def extract_text_from_pdf(pdf_path):
    """
    Extract all text from the PDF.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    return full_text

def clean_text(text):
    """
    Replace special/unusual characters with standard equivalents and normalize whitespace.
    For example:
      –  -> -
      —  -> -
      ’  -> '
      “ and ” -> '
    """
    replacements = {
        "–": "-",
        "—": "-",
        "’": "'",
        "“": "'",
        "”": "'",
        "\\":"",
        "☐": "",
        "©": "(c)",
        "™": "(tm)",
        "®": "(r)",
        "…": "...",
        "•": "-",
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    # Replace multiple whitespace characters with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_text_by_headings(full_text, headings):
    """
    Splits the full text using the specified headings.
    For each heading found in the text, the section text is all text from that heading
    until the next heading.
    
    Returns a dictionary mapping each heading to its corresponding text.
    """
    sections = {}
    # Build a regex pattern that matches any of the headings.
    # We use re.escape to escape any special characters in the headings.
    pattern = "(" + "|".join(re.escape(h) for h in headings) + ")"
    
    # Use re.finditer to get all heading matches with their positions.
    matches = list(re.finditer(pattern, full_text))
    if not matches:
        print("No headings found in the document.")
        return sections

    # For each match, slice the text from this heading to the next.
    for i, match in enumerate(matches):
        heading = match.group(0).strip()
        start_index = match.start()
        end_index = matches[i+1].start() if i+1 < len(matches) else len(full_text)
        section_text = full_text[start_index:end_index].strip()
        sections[heading] = section_text
    return sections

def main(pdf_path, output_json):
    print("Extracting text from PDF...")
    full_text = extract_text_from_pdf(pdf_path)
    print("Cleaning text...")
    full_text = clean_text(full_text)
    
    # List of headings exactly as they appear in the Table of Contents.
    headings = [
        # "Build and Maintain a Secure Network and Systems",
        "Requirement 1: Install and Maintain Network Security Controls",
        "Requirement 2: Apply Secure Configurations to All System Components",
        # "Protect Account Data",
        "Requirement 3: Protect Stored Account Data",
        "Requirement 4: Protect Cardholder Data with Strong Cryptography During Transmission Over Open, Public Networks",
        # "Maintain a Vulnerability Management Program",
        "Requirement 5: Protect All Systems and Networks from Malicious Software",
        "Requirement 6: Develop and Maintain Secure Systems and Software",
        # "Implement Strong Access Control Measures",
        "Requirement 7: Restrict Access to System Components and Cardholder Data by Business Need to Know",
        "Requirement 8: Identify Users and Authenticate Access to System Components",
        "Requirement 9: Restrict Physical Access to Cardholder Data",
        # "Regularly Monitor and Test Networks",
        "Requirement 10: Log and Monitor All Access to System Components and Cardholder Data",
        "Requirement 11: Test Security of Systems and Networks Regularly",
        # "Maintain an Information Security Policy",
        "Requirement 12: Support Information Security with Organizational Policies and Programs",
        #"Appendix A Additional PCI DSS Requirements",
        "A1 Additional PCI DSS Requirements for Multi-Tenant Service Providers",
        "A2 Additional PCI DSS Requirements for Entities Using SSL/Early TLS for Card-Present POS POI Terminal Connections",
        "A3 Designated Entities Supplemental Validation (DESV)",
        "Appendix B Compensating Controls",
        "Appendix C Compensating Controls Worksheet",
        "Appendix D Customized Approach",
        "Appendix E Customized Approach Template"
    ]
    
    print("Splitting document by headings...")
    sections = split_text_by_headings(full_text, headings)
    
    if not sections:
        print("No sections were extracted. Please check your headings list and document.")
        return
    
    print(f"Extracted {len(sections)} sections from the document.")
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

# import fitz  # PyMuPDF
# import json
# import re

# # Define the table of contents with section titles and their starting pages
# toc = [
#     ("Requirement 1: Install and Maintain Network Security Controls", 31),
#     ("Requirement 2: Apply Secure Configurations to All System Components", 62),
#     ("Requirement 3: Protect Stored Account Data", 82),
#     ("Requirement 4: Protect Cardholder Data with Strong Cryptography During Transmission Over Open, Public Networks", 129),
#     ("Requirement 5: Protect All Systems and Networks from Malicious Software", 138),
#     ("Requirement 6: Develop and Maintain Secure Systems and Software", 161),
#     ("Requirement 7: Restrict Access to System Components and Cardholder Data by Business Need to Know", 196),
#     ("Requirement 8: Identify Users and Authenticate Access to System Components", 216),
#     ("Requirement 9: Restrict Physical Access to Cardholder Data", 265),
#     ("Requirement 10: Log and Monitor All Access to System Components and Cardholder Data", 310),
#     ("Requirement 11: Test Security of Systems and Networks Regularly", 354),
#     ("Requirement 12: Support Information Security with Organizational Policies and Programs", 392),
#     ("Appendix A Additional PCI DSS Requirements", 449),
#     ("A1 Additional PCI DSS Requirements for Multi-Tenant Service Providers", 449),
#     ("A2 Additional PCI DSS Requirements for Entities Using SSL/Early TLS for Card-Present POS POI Terminal Connections", 459),
#     ("A3 Designated Entities Supplemental Validation (DESV)", 463),
#     ("Appendix B Compensating Controls", 464),
#     ("Appendix C Compensating Controls Worksheet", 466),
#     ("Appendix D Customized Approach", 467),
#     ("Appendix E Customized Approach Template", 469)
# ]

# def extract_text_by_sections(pdf_path, toc):
#     # Open the PDF document
#     doc = fitz.open(pdf_path)
#     sections = {}
    
#     # Iterate over each section in the table of contents
#     for i, (title, start_page) in enumerate(toc):
#         start_index = start_page - 1  # PyMuPDF uses 0-based indexing
#         end_index = toc[i + 1][1] - 2 if i + 1 < len(toc) else doc.page_count - 1
        
#         # Extract text from the specified page range
#         section_text = ""
#         for page_num in range(start_index, end_index + 1):
#             page = doc.load_page(page_num)
#             section_text += page.get_text()
        
#         # Clean the text by removing special characters
#         section_text = re.sub(r'[^\x00-\x7F]+', ' ', section_text)
#         section_text = section_text.replace('\\', '')
        
#         # Add the cleaned text to the sections dictionary
#         sections[title] = section_text.strip()
    
#     return sections

# def save_sections_to_json(sections, output_path):
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(sections, f, ensure_ascii=False, indent=4)

# # Usage
# pdf_path = 'C:/Users/nikit/OneDrive/Desktop/Panacea_Infosec/pci-roc-map/PCI-DSS-ROC-Template.pdf'
# output_json_path = 'output_sections.json'

# sections = extract_text_by_sections(pdf_path, toc)
# save_sections_to_json(sections, output_json_path)
