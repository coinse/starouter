import os
import argparse
import json
import time
import requests
import torch
import tqdm
import openai

endpoint = 'http://localhost:11434/api/embeddings'

def _query_model(payload):
    for _ in range(5):
        try:
            json_payload = json.dumps(payload)
            headers = {'Content-Type': 'application/json'}
            response = json.loads(requests.post(endpoint, data=json_payload, headers=headers).text)
            return response['embedding']
        except Exception as e:
            save_err = e
            if "The server had an error processing your request." in str(e):
                time.sleep(1)
            else:
                break
    raise save_err


def get_ollama_embeddings(initial_prompt):
    payload = {
        'model': 'nomic-embed-text',
        'prompt': initial_prompt,
        'stream': False
    }
    return _query_model(payload)

def get_openai_embeddings(text, model="text-embedding-3-large"):
    client = openai.OpenAI()
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def get_embeddings_from(text, model_type="nomic-embed-text"):
    if model_type == "nomic-embed-text":
        return get_ollama_embeddings(text)
    elif model_type == "text-embedding-3-large" or model_type == "text-embedding-3-small":
        return get_openai_embeddings(text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', default="nomic-embed-text")
    parser.add_argument('-b', '--benchmark', default="TestEval_line")
    parser.add_argument('--prompt_type', default="entire") 
    args = parser.parse_args()
    
    benchmark = args.benchmark.strip()
    model_type = args.type.strip()
    embeddings_path = f'{benchmark.lower()}/route_data/{model_type}_{args.prompt_type}_embeddings.pt'

    if os.path.isfile(embeddings_path):
        print(f"{embeddings_path} already exists!")
        exit()

    prompts_path = f'{benchmark.lower()}/route_data/{args.prompt_type}_prompts.json'
    embeddings_dict = dict()
    with open(prompts_path) as f:
        data = json.load(f)
    
    for bug_id in tqdm.tqdm(data):
        initial_prompt = data[bug_id]
        if type(initial_prompt) == list:
            initial_prompt = '\n'.join([str(msg) for msg in initial_prompt])
        embedding = get_embeddings_from(initial_prompt, model_type=model_type)
        embeddings_dict[bug_id] = embedding
    
    torch.save(embeddings_dict, embeddings_path)
