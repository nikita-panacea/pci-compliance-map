from PIL import Image
import os
import base64

from vllm import LLM
from vllm.sampling_params import SamplingParams

model_name = "mistralai/Pixtral-12B-2409"

sampling_params = SamplingParams(max_tokens=8192)

llm = LLM(model=model_name, tokenizer_mode="mistral")

def image_to_base64(file_path: str):
	with open(file_path, "rb") as image_file:
		encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
	_, extension = os.path.splitext(file_path)
	mime_type = f"image/{extension[1:].lower()}"
	
	return f"data:{mime_type};base64,{encoded_string}"

prompt = "Describe the image in detail."
image_file = "Connfido Network Diagram.png"
image_url = image_to_base64(image_file)
messages = [
	{
		"role":"user",
		"content":[{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": image_url}}]
	}
]

outputs = llm.chat(messages, sampling_params=sampling_params)
print(outputs[0].outputs[0].text)