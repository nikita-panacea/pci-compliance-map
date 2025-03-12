import os
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import (
    SystemMessage,
    UserMessage,
    TextContentItem,
    ImageContentItem,
    ImageUrl,
    ImageDetailLevel,
)
from azure.core.credentials import AzureKeyCredential

token = os.environ["GITHUB_TOKEN"]
endpoint = "https://models.inference.ai.azure.com"
model_name = "Phi-4-multimodal-instruct"
# model_name = "Phi-3.5-vision-instruct"
# model_name = "gpt-4o"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        UserMessage(
            content=[
                TextContentItem(text="""
        You are an expert in PCI-DSS compliance network security. You are provided with image evidence. 
        The image evidence has been provided by the audited organization as part of PCI DSS audit to prove 
        that they fulfil a particular control of PCI DSS framework.
        Provide a detailed report on the given image for compliance analysis:
        - List and name all components present in the image.
        - Describe the information presented in the image in detail.
        - Explain the information on security measures from the image.
        - Identify any potential vulnerabilities or security risks.
        """),
                ImageContentItem(
                    image_url=ImageUrl.load(
                        image_file="Connfido Network Diagram.png",
                        image_format="jpg",
                        detail=ImageDetailLevel.HIGH)
                ),
            ],
        ),
    ],
    model=model_name,
    temperature=1.0,
    top_p=1.0,
    max_tokens=2500,
)

print(response.choices[0].message.content)