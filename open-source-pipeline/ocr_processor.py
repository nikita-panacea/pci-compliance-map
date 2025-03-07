from PIL import Image
import easyocr
import json

class OCRProcessor:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
    
    def process_image(self, image_path, output_file="ocr_results.json"):
        # image = Image.open(image_path).convert("RGB")
        # if max(image.size) > 1024:
        #     image.thumbnail((1024, 1024))
        print("Extracting text from image...")
        result = self.reader.readtext(image_path, detail=0)
        print("Text extraction complete.")
        with open(output_file, "w") as f:
            json.dump({"text": "\n".join(result)}, f)
        return output_file

# usage
processor = OCRProcessor()
processor.process_image("C:/Users/nikit/OneDrive/Desktop/Panacea_Infosec/pci-roc-map/test_code/Connfido Network Diagram.png")