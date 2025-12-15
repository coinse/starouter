import argparse
import json
import os

import sys
sys.path.append("..")
from config import WEAK_MODEL_HUGGINGFACE_MAP, WEAK_MODEL_TOTAL_LAYERS_MAP

import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def extract_intermediate_states(tokenizer, model, prompt, layer_indices):
    if type(prompt) == str:
        messages = [{"role": "user", "content": prompt}]
    elif type(prompt) == list:
        messages = prompt
    else:
        raise Exception(f"Unsupported prompt type: {type(prompt)}")
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states

    extracted_vectors = {}
    for layer_idx in layer_indices:
        if 0 <= layer_idx < len(hidden_states):
            last_token_vector = hidden_states[layer_idx][:, -1, :].squeeze().cpu()
            extracted_vectors[layer_idx] = last_token_vector
        else:
            print(f"  - Warning: Layer {layer_idx} is out of range (Total layers: {len(hidden_states)-1}).")

    return extracted_vectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--benchmark', default='APPS')
    parser.add_argument('-m', '--model', default='llama3-8b')
    parser.add_argument('--prompt_type', default='entire')

    args = parser.parse_args()
    
    name = args.model
    layer_count = WEAK_MODEL_TOTAL_LAYERS_MAP[name]
    config = {"id": WEAK_MODEL_HUGGINGFACE_MAP[name], "layers": [round(layer_count / 2), round(layer_count * 2 / 3), round(layer_count * 3 / 4)]}

    output_path = f'{args.benchmark.lower()}/route_data/{name}_{args.prompt_type}_internal_states.pt'

    if os.path.isfile(output_path):
        print(f"{output_path} already exists!")
        exit()

    prompts_path = f'{args.benchmark.lower()}/route_data/{args.prompt_type}_prompts.json'
    embeddings_dict = dict()
    with open(prompts_path) as f:
        data = json.load(f)

    print(f"\n==============================================")
    print(f"Processing Model: {name}")
    print(config)
    print(f"==============================================")

    try:
        tokenizer = AutoTokenizer.from_pretrained(config['id'])
        model = AutoModelForCausalLM.from_pretrained(
            config['id'],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto"
        )
    except Exception as e:
        print(f"Failed to load model {config['id']}. Error: {e}")
        exit(1) 
    
    internal_states = dict()
    for id in tqdm.tqdm(data):
        prompt = data[id]
        try:
            vectors = extract_intermediate_states(
                tokenizer=tokenizer,
                model=model,
                prompt=prompt,
                layer_indices=config["layers"],
            )
            if vectors:
                internal_states[id] = vectors

        except torch.cuda.OutOfMemoryError:
            print(f"\nERROR: Ran out of VRAM for {name}. Try running models one by one.")
        except Exception as e:
            print(f"\nAn unexpected error occurred with {name}: {e}")

    torch.save(internal_states, output_path)
