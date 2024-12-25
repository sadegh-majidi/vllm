import requests
import json
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

VLLM_HOST = "http://0.0.0.0:8000"
url = f"{VLLM_HOST}/v1/completions"

# ANSI escape codes for colored output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

def send_prompt(prompt, num, model="meta-llama/Llama-2-7b-hf", max_tokens=700, temperature=0):
    start_time = time.time()
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
        response.raise_for_status()

        first_token_time = None
        for line in response.iter_lines():
            if line:
                first_token_time = time.time()
                decoded_line = line.decode("utf-8")
                result = json.loads(decoded_line).get("choices", [{}])[0].get("text", "No response text.")
                print(f"Prompt: {prompt}\n{Colors.OKGREEN}First Token: {result}{Colors.ENDC}\n")
                break

        if first_token_time:
            print(f"{Colors.OKCYAN}{num}. Time to first token: {first_token_time - start_time:.4f} seconds{Colors.ENDC}")
    except requests.RequestException as e:
        print(f"{Colors.FAIL}Error sending prompt '{prompt}': {e}{Colors.ENDC}")
    finally:
        end_time = time.time()
        print(f"{Colors.OKBLUE}{num}. Thread/Process finished for prompt: {prompt}. Start: {start_time:.4f}, End: {end_time:.4f}, Duration: {end_time - start_time:.4f} seconds{Colors.ENDC}")

def send_prompts_in_parallel(prompts, use_threads=True):
    n = min(len(prompts), 10)
    executor_class = ThreadPoolExecutor if use_threads else ProcessPoolExecutor

    with executor_class(max_workers=n) as executor:
        futures = [executor.submit(send_prompt, prompts[i], i) for i in range(n)]
        for future in futures:
            future.result()

def generate_prompts(batch_size, prompt_len):
    return ["A " * (prompt_len - 1) + "A" for _ in range(batch_size)]

if __name__ == "__main__":
    PROMPTS = generate_prompts(4, 1500)
    send_prompts_in_parallel(PROMPTS, use_threads=True)
