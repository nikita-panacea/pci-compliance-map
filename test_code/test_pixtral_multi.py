from PIL import Image
import os
import base64
from dotenv import load_dotenv
from mistralai import Mistral


mistral_api_key = 'WceN8y36a516WiLlxxjHcIUdKkEQhjFD'#os.getenv("MISTRAL_API_KEY")
model = "pixtral-12b-2409"
client = Mistral(api_key=mistral_api_key)

model_name = "mistralai/Pixtral-12B-2409"

def encode_image_base64(image_path): 
	with open(image_path, "rb") as image_file:   
		return base64.b64encode(image_file.read()).decode("utf-8")

def image_to_base64(file_path: str):
	with open(file_path, "rb") as image_file:
		encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
	_, extension = os.path.splitext(file_path)
	mime_type = f"image/{extension[1:].lower()}"
	
	return f"data:{mime_type};base64,{encoded_string}"

prompt = "Describe the image in detail."
image_file = 'C:/Users/nikit/OneDrive/Desktop/Panacea_Infosec/pci-roc-map/card_tokenization_flow.jpg'#"C:/Users/nikit/OneDrive/Desktop/Panacea_Infosec/pci-roc-map/card_decryption_flow.jpg"
image_url = encode_image_base64(image_file)

messages = [
	{
		"role":"user",
		"content":[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": f"data:image/png;base64,{image_url}"}]
	}
]

messages_2 = [
	{"role": "user", "content": [
		{"type": "text", "text": "According to the chart, how does Pixtral 12B performs compared to other models?"},
		{"type": "image_url", "image_url": "https://mistral.ai/images/news/pixtral-12b/pixtral-benchmarks.png"}
		]
	},
	]

#llm = LLM(model=model_name, tokenizer_mode="mistral")
chat_response = client.chat.complete(  
	model=model,  
	messages = messages)
print(chat_response.choices[0].message.content)
