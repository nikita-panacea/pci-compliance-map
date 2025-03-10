from openai import OpenAI
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
      
# test request
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, is this working?"}
    ]
)
        
# Check response
if response and response.choices[0].message.content:
    print("Response:", response.choices[0].message.content)
else:
    print("API response was empty or invalid")

 