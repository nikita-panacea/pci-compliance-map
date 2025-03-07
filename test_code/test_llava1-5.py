from transformers import pipeline 

# Load the LLaVA 1.5 model 
model_id = "llava-hf/llava-1.5-7b-hf" 
pipe = pipeline("image-to-text", model=model_id)  

# Provide an image URL
image_url = "Connfido Network Diagram.png" 

# Create a prompt for the image
prompt = f"""
You are an expert PCI-DSS compliance Auditor.
Describe this technical image in detail for compliance analysis:
- List all the devices and their roles in the image
- Explain the network topology and any security measures in place.
- Identify any potential vulnerabilities or security risks.
""" 

# Generate text description using LLaVA 
output = pipe(image=image_url, prompt=prompt) 

# Print the generated description 
print(output[0]["generated_text"]) 