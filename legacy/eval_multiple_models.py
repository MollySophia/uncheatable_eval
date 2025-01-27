import subprocess
import os
import time

os.environ['HF_HOME'] = './models/temp/'

data_list = [
    'data/ao3_english_20240501to20240515.json',
    'data/arxiv_computer_science_20240501to20240515.json',
    'data/arxiv_physics_20240501to20240515.json',
    'data/bbc_news_20240501to20240515.json',
    'data/github_cpp_20240501to20240515.json',
    'data/github_python_20240501to20240515.json',
    'data/wikipedia_english_20240501to20240515.json'
]
hf_cache = './models/temp/'

models = [
    'BlinkDL/rwkv-6-world/RWKV-x060-World-1B6-v2.1-20240328-ctx4096.pth',
    # 'BlinkDL/rwkv-5-world/RWKV-5-World-1B5-v2-20231025-ctx4096.pth',
    # 'stabilityai/stablelm-2-1_6b',
    # 'h2oai/h2o-danube-1.8b-base',
    # 'TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T',
    # 'EleutherAI/pythia-1.4b-v0',
    # 'bigscience/bloom-1b7',
    # 'BlinkDL/rwkv-4-pile-1b5/RWKV-4-Pile-1B5-20220903-8040.pth',
    # 'BlinkDL/rwkv-4-world/RWKV-4-World-1.5B-v1-fixed-20230612-ctx4096.pth',
    # 'allenai/OLMo-1B',
    # 'Qwen/Qwen1.5-1.8B',
    # 'Qwen/Qwen-1_8B',
    # 'princeton-nlp/Sheared-LLaMA-1.3B',
    # 'tiiuae/falcon-rw-1b',
    # 'OpenNLPLab/TransNormerLLM-1B',
    # 'microsoft/phi-1_5',
    # 'state-spaces/mamba-1.4b-hf',  # pip install causal-conv1d>=1.2.0 mamba-ssm, # use state-spaces/mamba-1.4b-hf instead of state-spaces/mamba-1.4b

    # 'BlinkDL/rwkv-6-world/RWKV-x060-World-3B-v2.1-20240417-ctx4096.pth',
    # 'BlinkDL/rwkv-5-world/RWKV-5-World-3B-v2-20231113-ctx4096.pth',
    # 'stabilityai/stablelm-3b-4e1t',
    # 'Qwen/Qwen1.5-4B',
    # 'state-spaces/mamba-2.8b-hf',  # use state-spaces/mamba-2.8b-hf instead of state-spaces/mamba-2.8b
    # 'openlm-research/open_llama_3b_v2',
    # 'cerebras/btlm-3b-8k-base',
    # 'state-spaces/mamba-2.8b-slimpj',
    # 'EleutherAI/pythia-2.8b-v0',
    # 'togethercomputer/RedPajama-INCITE-Base-3B-v1',
    # 'princeton-nlp/Sheared-LLaMA-2.7B',
    # 'BlinkDL/rwkv-4-pile-3b/RWKV-4-Pile-3B-20221110-ctx4096.pth',
    # 'BlinkDL/rwkv-4-world/RWKV-4-World-3B-v1-20230619-ctx4096.pth',

    # 'meta-llama/Meta-Llama-3-8B',
    # 'mistralai/Mistral-7B-v0.1',
    # 'BlinkDL/rwkv-5-world/RWKV-5-World-7B-v2-20240128-ctx4096.pth',
    # 'meta-llama/Llama-2-7b-hf',
    # 'TRI-ML/mamba-7b-rw',
    # 'Qwen/Qwen1.5-7B',
    # 'tiiuae/falcon-7b',
    # 'mosaicml/mpt-7b',
    # 'EleutherAI/pythia-6.9b-v0',
    # 'BlinkDL/rwkv-6-world/RWKV-x060-World-7B-v2.1-20240507-ctx4096.pth',
    # 'allenai/OLMo-1.7-7B-hf',
    # 'CohereForAI/aya-23-8B'
]

max_retries = 3

for model in models:
    if 'BlinkDL/rwkv-4-pile' in model:
        model_type = 'rwkv4pile'
    elif 'BlinkDL/rwkv' in model:
        model_type = 'rwkv'
    elif 'mamba' in model:
        model_type = 'mamba'
    else:
        model_type = 'hf'

    for data in data_list:
        if 'BlinkDL/rwkv' in model:
            command = f"echo 'y' | python3 eval_model.py --model {os.path.join(hf_cache, model.split('/')[-1])} --model_type {model_type} --data {data} --model_cache {hf_cache}"
        else:
            command = f"echo 'y' | python3 eval_model.py --model {model} --model_type {model_type} --data {data} --model_cache {hf_cache}"

        for attempt in range(max_retries):
            try:
                result = subprocess.run(command, shell=True, text=True, check=True)
                if result.returncode == 0:
                    # print(f"Command succeeded on attempt {attempt + 1}")
                    break
            except subprocess.CalledProcessError as e:
                print(f"Command failed on attempt {attempt + 1} with error: {e}")
                if attempt == max_retries - 1:
                    print("Max retries reached. Command failed.")
                else:
                    time.sleep(10)
