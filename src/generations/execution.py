import google.generativeai as genai 
import anthropic
import os
import PIL.Image 
import base64

def run_gemini(prompt, model_id="gemini-1.5-flash"):
    client = genai.GenerativeModel(model_name=model_id)
    return client.generate_content([prompt]).text

def visual_gemini_run(prompt, image_path, model_id="gemini-1.5-flash"):
    client = genai.GenerativeModel(model_name=model_id)
    img = PIL.Image.open(image_path)
    prompt = "You are given a control flow graph image of a code snippet, utilize them in code execution reasoning process. " + prompt
    return client.generate_content([img, prompt]).text

def anthropic_run(prompt, model_id="claude-3-5-sonnet-20240620"):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))  
    response = client.messages.create(
        model=model_id,
        system="You are a helpful assistant.",
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=2048
    )
    return response.content[0].text

def visual_anthropic_run(prompt, image_path, model_id="claude-3-5-sonnet-20240620"):
    with open(image_path, 'rb') as f:
        image_data = f.read()
    image_data = base64.b64encode(image_data).decode('utf-8')

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))  
    response = client.messages.create(
        model=model_id,
        system="You are a helpful assistant.",
        messages=[
        { "role": "user", "content": [  
            { 
                "type": "text", 
                "text": "You are given a control flow graph image of a code snippet, utilize them in code execution reasoning process. " + prompt, 
            },
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_data
                }
            },
        ] }
    ],
        max_tokens=2048 
    )

    return response.content[0].text

def run_model_execution(prompt, api_type, model_id, visual_aid: bool=False, image_path=None):
    if api_type == "gemini":
        if visual_aid:
            return visual_gemini_run(prompt, image_path, model_id)
        return run_gemini(prompt, model_id)
    if api_type == "anthropic":
        if visual_aid:
            return visual_anthropic_run(prompt, image_path, model_id)
        return anthropic_run(prompt, model_id)
    if api_type == "internvl2":
        pass
    
    raise ValueError("Invalid API type")