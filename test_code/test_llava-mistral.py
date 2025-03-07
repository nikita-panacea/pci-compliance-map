import ollama
import easyocr
import pytesseract

image_path = "Connfido Network Diagram.png"
image_ocr = pytesseract.image_to_string(image_path)
# general_ocr = easyocr.Reader(['en'], gpu=False)
# image_ocr = general_ocr.readtext(image_path, detail=0)
prompt = f"""
You are an expert PCI-DSS compliance Auditor.
Describe this technical image in detail for compliance analysis:
- List all the devices and their roles in the image
- Explain the network topology and any security measures in place.
- Identify any potential vulnerabilities or security risks.
Context text:
{image_ocr}
"""

# Visual understanding with LLaVA
response = ollama.generate(
    model="llava:7b-v1.6-mistral-q4_K_M",
    prompt=prompt,
    images=[image_path]
)
visual_description = response['response']

print(visual_description)