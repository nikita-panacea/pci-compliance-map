import os
import re
import json
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    Extracts full text from the PDF using PyMuPDF.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") + "\n"
    return full_text

def clean_text(text):
    """
    Replaces special characters with standard equivalents and normalizes whitespace.
    For example:
      –  -> -
      ’  -> '
    Also removes extra spaces and line breaks.
    """
    replacements = {
        "–": "-",
        "—": "-",
        "’": "'",
        "“": '"',
        "”": '"',
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    # Replace multiple whitespace/newlines with a single space
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def get_heading_list():
    """
    Returns an ordered list of heading titles extracted from the Table of Contents.
    (These should be exactly as they appear in the TOC.)
    """
    headings = [
        "ROC Template Instructions",
        "ROC Sections",
        "Assessment Findings",
        "What Is the Difference between Not Applicable and Not Tested?",
        "Dependence on Another Service Provider’s Compliance",
        "Assessment Approach Reporting Options",
        "Understanding the Reporting Instructions",
        "Dos and Don’ts: Reporting Expectations",
        "PCI DSS v4.0 Report on Compliance Template",
        "Part I Assessment Overview",
        "1 Contact Information and Summary of Results",
        "1.1 Contact Information",
        "1.2 Date and Timeframe of Assessment",
        "1.3 Remote Assessment Activities",
        "1.3.1 Overview of Remote Testing Activity",
        "1.3.2 Summary of Testing Performed Remotely",
        "1.3.3 Assessor Assurance in Assesment Result",
        "1.3.4 Requirements That Could Not be Fully Verified",
        "1.4 Additional Services Provided by QSA Company",
        "1.5 Use of Subcontractors",
        "1.6 Additional Information/Reporting",
        "1.7 Overall Assessment Result",
        "1.8 Summary of Assessment",
        "1.8.1 Summary of Assessment Findings and Methods",
        "1.8.2 Optional: Additional Assessor Comments",
        "1.9 Attestation Signatures",
        "2 Business Overview",
        "2.1 Description of the Entity’s Payment Card Business",
        "3 Description of Scope of Work and Approach Taken",
        "3.1 Assessor’s Validation of Defined Scope Accuracy",
        "3.2 Segmentation",
        "3.3 PCI SSC Validated Products and Solutions",
        "3.4 Sampling",
        "4 Details About Reviewed Environments",
        "4.1 Network Diagrams",
        "4.2 Account Dataflow Diagrams",
        "4.2.1 Description of Account Data Flows",
        "4.3 Storage of Account Data",
        "4.3.1 Storage of SAD",
        "4.4 In-scope Third-Party Service Providers (TPSPs)",
        "4.5 In-scope Networks",
        "4.6 In-scope Locations/Facilities",
        "4.7 In-scope Business Functions",
        "4.8 In-scope System Component Types",
        "4.9 Sample Sets for Reporting",
        "5 Quarterly Scan Results",
        "5.1 Quarterly External Scan Results",
        "5.2 Attestations of Scan Compliance",
        "5.3 Quarterly Internal Scan Results",
        "6 Evidence (Assessment Workpapers)",
        "6.1 Evidence Retention",
        "6.2 Documentation Evidence",
        "6.3 Interview Evidence",
        "6.4 Observation Evidence",
        "6.5 System Evidence",
        "Part II Findings and Observations",
        "Build and Maintain a Secure Network and Systems",
        "Requirement 1: Install and Maintain Network Security Controls",
        "Requirement 2: Apply Secure Configurations to All System Components",
        "Protect Account Data",
        "Requirement 3: Protect Stored Account Data",
        "Requirement 4: Protect Cardholder Data with Strong Cryptography During Transmission Over Open, Public Networks",
        "Maintain a Vulnerability Management Program",
        "Requirement 5: Protect All Systems and Networks from Malicious Software",
        "Requirement 6: Develop and Maintain Secure Systems and Software",
        "Implement Strong Access Control Measures",
        "Requirement 7: Restrict Access to System Components and Cardholder Data by Business Need to Know",
        "Requirement 8: Identify Users and Authenticate Access to System Components",
        "Requirement 9: Restrict Physical Access to Cardholder Data",
        "Regularly Monitor and Test Networks",
        "Requirement 10: Log and Monitor All Access to System Components and Cardholder Data",
        "Requirement 11: Test Security of Systems and Networks Regularly",
        "Maintain an Information Security Policy",
        "Requirement 12: Support Information Security with Organizational Policies and Programs",
        "Appendix A Additional PCI DSS Requirements",
        "A1 Additional PCI DSS Requirements for Multi-Tenant Service Providers",
        "A2 Additional PCI DSS Requirements for Entities Using SSL/Early TLS for Card-Present POS POI Terminal Connections",
        "A3 Designated Entities Supplemental Validation (DESV)",
        "Appendix B Compensating Controls",
        "Appendix C Compensating Controls Worksheet",
        "Appendix D Customized Approach",
        "Appendix E Customized Approach Template"
    ]
    # Normalize headings by stripping extra whitespace and converting special quotes to standard ones.
    normalized = [h.strip().replace("’", "'") for h in headings]
    return normalized

def split_document_by_toc(full_text, heading_list):
    """
    Splits the full text using only the heading titles in heading_list as boundaries.
    The function searches for each heading in order and extracts the text until the next heading.
    Returns a dictionary mapping each heading to its content.
    """
    # Pre-clean the text.
    clean = clean_text(full_text)
    
    sections = {}
    # Use regex to find each heading position.
    pattern = "(" + "|".join([re.escape(h) for h in heading_list]) + ")"
    # The regex is case-sensitive; if needed, set re.IGNORECASE.
    splits = re.split(pattern, clean)
    # re.split with a capturing group returns a list where headings are preserved.
    # Expect the list to be: [pre_text, heading, text, heading, text, ...]
    # We ignore pre_text (if any) and use the headings as keys.
    current_heading = None
    for i, part in enumerate(splits):
        part = part.strip()
        if part in heading_list:
            current_heading = part
            sections[current_heading] = ""
        else:
            if current_heading:
                # Append the text to current heading.
                sections[current_heading] += part + " "
    # Optionally, remove empty sections.
    sections = {k: v.strip() for k, v in sections.items() if v.strip()}
    return sections

def main(pdf_path, output_json):
    print("Extracting full text from PDF...")
    full_text = extract_text_from_pdf(pdf_path)
    print("Cleaning text...")
    full_text = clean_text(full_text)
    print("Splitting document using Table of Contents headings...")
    headings = get_heading_list()
    sections = split_document_by_toc(full_text, headings)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(sections, f, ensure_ascii=False, indent=2)
    print(f"Processed PDF and saved sections to {output_json}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python process_pci_pdf.py <path_to_pdf> <output_json>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    output_json = sys.argv[2]
    main(pdf_path, output_json)
