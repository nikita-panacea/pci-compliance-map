import ollama
import json

image_path = "/home/omen/Documents/Nikita/pci-compliance-map/Connfido Network Diagram.png"
# Load ocr file:
with open("ocr_results.json", "r") as f:
    image_ocr = json.load(f)["text"]

# Visual understanding with LLaVA
prompt = f"""
You are an expert in PCI-DSS compliance network. security. 
Provide a detailed report on the given image.
Describe this image in detail, using at least a 1000 words or more, for compliance analysis:
- List all the systems and devices present and their roles in the image.
- List all connections present explaining network architecture.
- Explain the network topology and any security measures in place.
- Identify any potential vulnerabilities or security risks.
Context text:
{image_ocr}
"""

class VisualAnalyzer:
    def __init__(self):
        self.model_name = "minicpm-v:latest" # llama3.2-vision:latest
        
    def analyze(self, image_path, output_file="vision_analysis_minicpm.json"):
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