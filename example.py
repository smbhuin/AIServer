from openai import OpenAI

client = OpenAI(api_key="demo", base_url="http://localhost:8000/v1")

def chat_completion():
    completion = client.chat.completions.create(
    model="mistral-7b",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"}
    ]
    )

    print(completion.choices[0].message.content)

def image_generation():
    response = client.images.generate(
        model="dreamshaper-8",
        prompt="portrait photo of muscular bearded guy in a worn mech suit, light bokeh, intricate, steel metal, elegant, sharp focus, soft lighting, vibrant colors",
        size="512x512",
        quality="standard",
        n=1,
    )

    print(response.data[0].url)


#chat_completion()
image_generation()