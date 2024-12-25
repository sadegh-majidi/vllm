import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

VLLM_HOST = "http://0.0.0.0:8000"
url = f"{VLLM_HOST}/v1/completions"

def send_prompt(prompt, num, model="meta-llama/Llama-2-7b-hf", max_tokens=700, temperature=0):
    start_time = time.time()
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        result = response.json().get("choices", [{}])[0].get("text", "No response text.")
        print(f"Prompt: {prompt}\nResponse: {result}\n")
    except requests.RequestException as e:
        print(f"Error sending prompt '{prompt}': {e}")
    finally:
        end_time = time.time()
        print(f"{num}. Thread/Process finished for prompt: {prompt}. Start: {start_time} End: {end_time} Duration: {end_time - start_time}")

PROMPTS = [
    "This is",
    "Emily said",
    "Cops arrived",
    "I pushed",
    "Death is",
    "Please stay",
    "Arthur remorsefully",
    "Whatever happens",
    "Though he",
    "Love hurts"
]

def send_prompts_in_parallel(prompts, use_threads=True):
    n = min(len(prompts), 10)
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with executor_class(max_workers=n) as executor:
        futures = [executor.submit(send_prompt, prompts[i], i) for i in range(n)]
        for future in futures:
            future.result()

if __name__ == "__main__":
    send_prompts_in_parallel(PROMPTS, use_threads=True)
