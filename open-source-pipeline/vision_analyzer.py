import ollama
import json

image_path = "C:/Users/nikit/OneDrive/Desktop/Panacea_Infosec/pci-roc-map/test_code/Connfido Network Diagram.png"
# Load ocr file:
with open("ocr_results.json", "r") as f:
    image_ocr = json.load(f)["text"]

# Visual understanding with LLaVA
prompt = f"""
You are an expert PCI-DSS compliance Auditor.
Describe this technical image in detail for compliance analysis:
- List all the devices and their roles in the image
- Explain the network topology and any security measures in place.
- Identify any potential vulnerabilities or security risks.
Context text:
{image_ocr}
"""

class VisualAnalyzer:
    def __init__(self):
        self.model_name = "llava:7b-v1.6-mistral-q4_K_M"
        
    def analyze(self, image_path, output_file="vision_analysis.json"):
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

# usage
analyzer = VisualAnalyzer()
analyzer.analyze(image_path)