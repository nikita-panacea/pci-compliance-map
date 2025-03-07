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
# model_name = "Phi-4-multimodal-instruct"
# model_name = "Phi-3.5-vision-instruct"
model_name = "gpt-4o"

client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

response = client.complete(
    messages=[
        UserMessage(
            content=[
                TextContentItem(text="You are an Network security expert. Write a analysis report explaining the data flow."),
                ImageContentItem(
                    image_url=ImageUrl.load(
                        image_file="card_decryption_flow.jpg",
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