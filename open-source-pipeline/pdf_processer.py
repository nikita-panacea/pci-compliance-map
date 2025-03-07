import fitz
import json
import re
from typing import Dict, List

class PCIProcessor:
    def __init__(self):
        self.header_pattern = re.compile(
            r'^(Part [I]+|Requirement \d+|Appendix [A-Z]|\d+\.\d+(\.\d+[a-z]?)?)',
            re.IGNORECASE
        )
        self.current_section = None
        self.current_subsection = None

    def process_pdf(self, pdf_path: str) -> Dict:
        """Improved PDF processor with better error handling"""
        doc = fitz.open(pdf_path)
        structured_data = self._create_initial_structure()
        content_buffer = []
        
        for page in doc:
            text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            
            for line in text.split('\n'):
                line = line.strip()
                if self._is_section_header(line):
                    if content_buffer:
                        self._save_content(structured_data, content_buffer)
                        content_buffer = []
                    self._update_section_hierarchy(line)
                else:
                    content_buffer.append(line)
        
        if content_buffer:
            self._save_content(structured_data, content_buffer)
        
        with open("pci_structured.json", "w") as f:
            json.dump(structured_data, f, indent=2)
        
        return structured_data

    def _is_section_header(self, line: str) -> bool:
        """Check if line matches known section patterns"""
        return bool(self.header_pattern.match(line))

    def _update_section_hierarchy(self, line: str):
        """Handle section hierarchy without index errors"""
        match = self.header_pattern.search(line)
        if not match:
            return
            
        header = match.group(1)
        parts = header.split()
        
        try:
            if parts[0].lower() == "part":
                self.current_section = header
                self.current_subsection = None
            elif parts[0].lower() == "requirement":
                self.current_section = header.replace(":", "")
                self.current_subsection = None
            elif parts[0].lower() == "appendix":
                self.current_section = header
                self.current_subsection = None
            elif re.match(r'\d+\.\d+', parts[0]):
                self.current_subsection = parts[0]
        except IndexError:
            pass

    def _save_content(self, data: Dict, buffer: List[str]):
        """Safer content saving with error handling"""
        if not self.current_section:
            return
            
        content = "\n".join(buffer)
        section_type = self._get_section_type()
        
        try:
            if section_type == "requirement":
                req_num = self.current_section.split()[-1]
                data["requirements"].setdefault(req_num, {
                    "text": content,
                    "subsections": {}
                })
            elif section_type == "appendix":
                data["appendices"].append({
                    "title": self.current_section,
                    "content": content
                })
            else:
                data["other_sections"][self.current_section] = content
        except KeyError as e:
            print(f"Warning: Failed to save content to section {self.current_section}: {e}")

    def _get_section_type(self) -> str:
        """Identify section type safely"""
        if not self.current_section:
            return "other"
            
        if "Requirement" in self.current_section:
            return "requirement"
        if "Appendix" in self.current_section:
            return "appendix"
        if "Part" in self.current_section:
            return "part"
        return "other"

    def _create_initial_structure(self) -> Dict:
        return {
            "metadata": {
                "document_type": "PCI-DSS ROC",
                "version": "4.0"
            },
            "requirements": {},
            "appendices": [],
            "other_sections": {}
        }

    def _save_structured_data(self, structured_data: Dict, output_file: str):
        with open(output_file, "w") as f:
            json.dump(structured_data, f, indent=2)

if __name__ == "__main__":
    processor = PCIProcessor()
    processor.process_pdf("C:/Users/nikit/OneDrive/Desktop/Panacea_Infosec/pci-roc-map/PCI-DSS-ROC-Template.pdf")
    