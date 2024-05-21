from django.shortcuts import render
from huggingface_hub import InferenceClient
import random, time


API_URL = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
headers = {"Authorization": "Bearer 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"}



import requests


def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
output = query({
	"inputs": "Can you please let us know more details about your ",
})


# Create your views here.
def home(request):
    if request.method == 'POST':
        x = request.body
        print(type(x))
        print(x)
        x = x.decode('utf-8')
        print(x)
        x = x.split('messageText=')
        print(x)
        
        user_prompt = x[1]
        generated_text = generate(user_prompt)
    
        print("Mistral7B: ", generated_text)
        
        
        
        data = {
            'response' : generated_text
        }
        return render(request, "chat/chat.html", data)
    else:
        return render(request, "chat/chat.html")
    




def format_prompt(message, custom_instructions=None):
    prompt = ""
    if custom_instructions:
        prompt += f"[INST]{custom_instructions}[/INST]"

    if message:
        prompt += f"[INST]{message}[/INST]"
        
    return prompt


def generate(prompt, temperature=0.9, max_new_tokens=512, top_p=0.95, repetition_penalty=1.0):
    temperature = float(temperature)
    if temperature < 1e-2:
        temperature = 1e-2
        
    top_p = float(top_p)
    
    generate_kwargs = dict(
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        do_sample=True,
        seed=random.randint(0, 10**7)
    )
    
    custom_instructions = "Be professional."
    formatted_prompt = format_prompt(prompt, custom_instructions)
    
    #model call
    client = InferenceClient(API_URL, headers=headers)
    
    response = client.text_generation(formatted_prompt, **generate_kwargs)
    
    return response
    
    