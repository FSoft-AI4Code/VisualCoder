from __future__ import annotations
from ast import arg
import base64
import json
from tqdm import tqdm
from src.utils.utils import write
from src.codetransform.cfg2image import code2cfgimage
import re
import os
import src.utils.options
import anthropic
from openai import AzureOpenAI 
import requests

args = src.utils.options.options().parse_args()

# Load dataset
with open("dataset.json", "r") as f:
    dataset = json.load(f)

pattern = r"```python(.*?)```"

def get_buggy_cfg_COT_prompt(code: str):
    SYSTEM_MESSAGE = """You are an expert Python programmer."""

    prompt = f"""Analyze the following Python code snippet, which contains error(s) when executing:

{code}

You will also be provided with a control flow graph (CFG) image of this code. As you analyze each line:
1. Examine each line of code sequentially to understand the overall structure and purpose of the code using both the code and CFG.
2. Use this understanding to support your reasoning about the code's logic and potential errors.

Think through your analysis step by step, considering both the code and its representation in the CFG.

After your analysis, respond with the line number and the problematic line of code that causes termination, separated by a colon. For example: '3: problematic_line_of_code_here'."""

    return prompt, SYSTEM_MESSAGE

def get_VisualCoder_prompt(code: str):
    SYSTEM_MESSAGE = """You are an expert Python programmer."""

    prompt = f"""Analyze the following Python code snippet, which contains error(s) when executing:

{code}

You will also be provided with a control flow graph (CFG) image of this code. As you analyze each line:
1. Examine each line of code sequentially.
2. Reference the CFG to identify which node corresponds to the line you're currently analyzing.
3. Use this alignment between code and CFG to support your reasoning about the code's logic and potential errors.

Think through your analysis step by step, considering both the code and its representation in the CFG.

After your analysis, respond with the line number and the problematic line of code that causes termination, separated by a colon. For example: '3: problematic_line_of_code_here'."""

    return prompt, SYSTEM_MESSAGE

def get_buggy_COT_prompt(code: str):
    SYSTEM_MESSAGE = """You are an expert Python programmer."""

    prompt = f"""Analyze the following Python code snippet, which contains error(s) when executing:

{code}

Please follow these steps in your analysis:
1. Examine each line of code sequentially to understand code.
2. Use this understanding to support your reasoning about the code's logic and potential errors.

Think through your analysis step by step, considering the code's logic and potential errors.

After your analysis, respond with the line number and the problematic line of code that causes termination, separated by a colon. For example: '3: problematic_line_of_code_here'"""

    return prompt, SYSTEM_MESSAGE

def get_buggy_prompt(code: str):
    SYSTEM_MESSAGE = """You are an expert Python programmer."""
    prompt = f"Here's a Python code snippet with error(s):\n\n```python\n{code}\n```\n\n"
    prompt += "Please identify the specific line causing code terminate. Respond with the line number and the problematic line of code that causes termination, separated by a colon. For example: '3: problematic_line_of_code_here'"
    return prompt, SYSTEM_MESSAGE

def get_buggy_cfg_prompt(code: str):
    SYSTEM_MESSAGE = """You are an expert Python programmer. You will be given a Python code snippet and its control flow graph image. The code contains error(s). Your task is identify the specific line in the code that make the code terminate. Please respond with the line number and the problematic line of code, separated by a colon. For example: '3: problematic_line_of_code_here'"""
    prompt = f"Here's a Python code snippet with error(s):\n\n```python\n{code}\n```\n\n"
    prompt += "Please identify the specific line causing code terminate. Respond with the line number and the problematic line of code that causes termination, separated by a colon. For example: '3: problematic_line_of_code_here'"
    return prompt, SYSTEM_MESSAGE

def get_rationale_generation_prompt(code: str):
    SYSTEM_MESSAGE = """You are an expert Python programmer."""

    prompt = f"""Analyze the following Python code snippet, which contains error(s) when executing:

{code}

You will also be provided with a control flow graph (CFG) image of this code if applicable. As you analyze each line:
1. Examine each line of code sequentially.
2. Reference the CFG to identify which node corresponds to the line you're currently analyzing.
3. Use this alignment between code and CFG to generate a rationale for the error-prone code.

After your analysis, provide a detailed rationale explaining what might be wrong with the code."""

    return prompt, SYSTEM_MESSAGE

def get_rationale_generation_VisualCoder_prompt(code: str):
    SYSTEM_MESSAGE = """You are an expert Python programmer."""

    prompt = f"""Analyze the following Python code snippet, which contains error(s) when executing:

{code}

You will also be provided with a control flow graph (CFG) image of this code if applicable. As you analyze each line:
1. Examine each line of code sequentially.
2. Reference the CFG to identify which node corresponds to the line you're currently analyzing.
3. Use this alignment between code and CFG to generate a rationale for the error-prone code.

After your analysis, provide a detailed rationale explaining what might be wrong with the code."""

    return prompt, SYSTEM_MESSAGE
def get_answer_inference_prompt(code: str, rationale: str):
    SYSTEM_MESSAGE = """You are an expert Python programmer."""

    prompt = f"""You have a Python code snippet containing error(s) and a rationale for the error(s).

Code:

{code}

Rationale: {rationale}

Using this rationale, please identify the specific line of code that causes termination. Respond with the line number and the problematic line of code that causes termination, separated by a colon. For example: '3: problematic_line_of_code_here'."""

    return prompt, SYSTEM_MESSAGE


def code_to_image(path: str):
    with open(path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')

def run_error_detection(code, image_path, generation_config):
    global openai_api_key
    if args.setting == "buggy_COT":
        prompt, SYSTEM_MESSAGE = get_buggy_COT_prompt(code)
    elif args.setting == "buggy_cfg_COT":
        prompt, SYSTEM_MESSAGE = get_buggy_cfg_COT_prompt(code)
    elif args.setting == "buggy":
        prompt, SYSTEM_MESSAGE = get_buggy_prompt(code)
    elif args.setting == "buggy_cfg":
        prompt, SYSTEM_MESSAGE = get_buggy_cfg_prompt(code)
    elif args.setting == "VisualCoder":
        prompt, SYSTEM_MESSAGE = get_VisualCoder_prompt(code)

    if 'cfg' in args.setting or args.setting == "VisualCoder":
        if args.close_model == "claude":
            client = anthropic.Anthropic(api_key=args.claude_api_key)
            image_data = code_to_image(image_path)
            chat_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": image_data
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                },
            ]
            response = client.messages.create(
                model=args.version,
                system=SYSTEM_MESSAGE,
                messages=chat_messages,
                max_tokens=1024
            )
            response_text = response.content[0].text.strip()
        elif args.close_model == "gpt":
            image_data = code_to_image(image_path)
            headers = {
                "Content-Type": "application/json",
                "api-key": args.openai_api_key
            }

            payload = {
                "model": args.version,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_MESSAGE,
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ]
                    }
                ],
                "max_tokens": 2048
            }

            response = requests.post(
                args.azure_endpoint,
                headers=headers,
                json=payload
            )
            response_text = response.json()["choices"][0]["message"]["content"]
    else:
        if args.close_model == "claude":
            client = anthropic.Anthropic(api_key=args.claude_api_key)
            chat_messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                },
            ]
            response = client.messages.create(
                model=args.version,
                system=SYSTEM_MESSAGE,
                messages=chat_messages,
                max_tokens=1024
            )
            response_text = response.content[0].text.strip()
        elif args.close_model == "gpt":
            client = AzureOpenAI(
                azure_endpoint=args.azure_endpoint,
                api_key=args.openai_api_key,
                api_version="2024-02-01",
                azure_deployment=args.deployment_name
                )
            chat_messages = [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            response = client.chat.completions.create(
                model=args.version,
                messages=chat_messages,
                max_tokens=1024
            )
            response_text = response.choices[0].message.content
    return response_text

def run_error_detection_two_stage(code, image_path, generation_config):
    global openai_api_key
    if args.setting == "Multimodal-CoT":
       prompt_stage1, SYSTEM_MESSAGE = get_rationale_generation_prompt(code)
    elif args.setting == "Multimodal-CoT_VisualCoder":
        prompt_stage1, SYSTEM_MESSAGE = get_rationale_generation_VisualCoder_prompt(code)
    
    image_data = code_to_image(image_path) 
    rationale = ""
    if args.close_model == "gpt":
        headers = {
            "Content-Type": "application/json",
            "api-key": openai_api_key
        }
        payload = {
            "model": args.version,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE,
                },
                {
                    "role": "user",
                    "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_data}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt_stage1,
                            }
                        ]
                }
            ],
            "max_tokens": 2048
        }

        response = requests.post(
            args.azure_endpoint,
            headers=headers,
            json=payload
        )

        rationale = response.json()["choices"][0]["message"]["content"]

    elif args.close_model == "claude":
        client = anthropic.Anthropic(api_key=args.claude_api_key)  # Use the actual API key here

        chat_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt_stage1
                    }
                ]
            },
        ]

        response = client.messages.create(
            model=args.version,
            system=SYSTEM_MESSAGE,
            messages=chat_messages,
            max_tokens=1024
        )
        rationale = response.content[0].text
    
    # Stage 2: Answer Inference
    prompt_stage2, SYSTEM_MESSAGE = get_answer_inference_prompt(code, rationale)
    
    if args.close_model == "gpt":
        payload = {
            "model": args.version,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE,
                },
                {
                    "role": "user",
                    "content": prompt_stage2
                }
            ],
            "max_tokens": 2048
        }
        
        response = requests.post(
            args.azure_endpoint,
            headers=headers,
            json=payload
        )

        response_text = response.json()["choices"][0]["message"]["content"]

    elif args.close_model == "claude":
        chat_messages = [
            {
                "role": "user",
                "content": prompt_stage2
            }
        ]
        response = client.messages.create(
            model=args.version,
            system=SYSTEM_MESSAGE,
            messages=chat_messages,
            max_tokens=1024
        )
        response_text = response.content[0].text
    
    return rationale, response_text

results = []
path = f"output/bug_localization/{args.close_model}"
if not os.path.exists(path):
    os.makedirs(path)
path = os.path.join(path, f"{args.setting}_{args.session}.txt")

path = "output/bug_localization/analysis.txt"
write(f"Setting: {args.setting}", path)
for idx in tqdm(range(len(dataset))):
    data_point = dataset[idx]
    code = data_point["code"]
    actual_error_line = data_point["error_line"]
    if 'cfg' in args.setting or args.setting == "Multimodal-CoT" or args.setting == "Multimodal-CoT_VisualCoder" or args.setting == "VisualCoder":
        filepath, _ = code2cfgimage(code, filename=f"{args.setting}_{args.session}")
    else:
        filepath = None

    generation_config = dict(max_new_tokens=1024, do_sample=False)
    write(f"Problem {idx}", path)
    if args.setting == "Multimodal-CoT" or args.setting == "Multimodal-CoT_VisualCoder":
        rationale, response = run_error_detection_two_stage(code, filepath, generation_config)
        write(f"Rationale: {rationale}", path)
        write("--------->", path)
    else:
        response = run_error_detection(code, filepath, generation_config)
    # split the response into each line by \n, find the line that start with a number
    response_lines = response.split("\n")
    result_line = ""
    for line in response_lines:
        if len(line.strip()) > 0 and ":" in line:
            if line[0] == "'" and line[-1] == "'":
                line = line[1:-1]
            if line[0] == "`" and line[-1] == "`":
                line = line[1:-1]
            if line[0] == "'" and line[-1] == "'":
                line = line[1:-1]
            if line[0] == "``" and line[-1] == "``":
                line = line[2:-2]
            if line[0] == "```" and line[-1] == "```":
                line = line[3:-3]
            if line[0] == "''" and line[-1] == "''":
                line = line[2:-2]
            if line[0] == "'''" and line[-1] == "'''":
                line = line[3:-3]   
            if line[0] == "**" and line[-1] == "**":
                line = line[2:-2]
            if line[0] == "***" and line[-1] == "***":
                line = line[3:-3]     
            if line.split(":")[0].strip().isdigit():
                result_line = line
                break
    if result_line == "":
        # detect all p
        for i in range(len(response_lines)):
            if "problematic" in response_lines[i]:
                response = '\n'.join(response_lines[i:])
                break
        match = re.findall(pattern, response, re.DOTALL)
        if match != []:
            predicted_error_code = ""
            for line in match:
                if line.strip().count("\n") == 0:
                    predicted_error_code = line.strip()
                    break
            is_correct = (predicted_error_code.replace(" ", "") == actual_error_line.replace(" ", ""))
            result = {
            "index": idx,
            "predicted": predicted_error_code,
            "actual": actual_error_line,
            "is_correct": is_correct
            }
            results.append(result)
            write(f"{response}", path)
            write(f"Predicted: {predicted_error_code}", path)
            write(f"Actual: {actual_error_line}", path)
            write(f"Is Correct: {is_correct}", path)
            write("--------------------", path)
            continue
        else:
            result_line = "1: No valid answer"

    write(f"{response}", path)
    predicted_error_line = result_line.split(":")[0].strip()
    predicted_error_code = ":".join(result_line.split(":")[1:]).strip()
    predicted_error_code = predicted_error_code.split("\n")[0]
    if len(predicted_error_code) > 0:
        if predicted_error_code[0] == "`" and predicted_error_code[-1] == "`":
            predicted_error_code = predicted_error_code[1:-1]
        if predicted_error_code[0] == "'" and predicted_error_code[-1] == "'":
            predicted_error_code = predicted_error_code[1:-1]
        if predicted_error_code[0] == "``" and predicted_error_code[-1] == "``":
            predicted_error_code = predicted_error_code[2:-2]
        if predicted_error_code[0] == "```" and predicted_error_code[-1] == "```":
            predicted_error_code = predicted_error_code[3:-3]
        if predicted_error_code[0] == "''" and predicted_error_code[-1] == "''":
            predicted_error_code = predicted_error_code[2:-2]
        if predicted_error_code[0] == "'''" and predicted_error_code[-1] == "'''":
            predicted_error_code = predicted_error_code[3:-3]    
    lines = [element.lstrip() for element in code.split("\n")]
    if int(predicted_error_line) > len(lines):
        is_correct = (predicted_error_code.replace(" ", "") == actual_error_line.replace(" ", ""))
        result = {
        "index": idx,
        "predicted": code_line,
        "actual": actual_error_line,
        "is_correct": is_correct
        }
        results.append(result)
        
        write(f"Predicted: {predicted_error_code}", path)
        write(f"Actual: {actual_error_line}", path)
        write(f"Is Correct: {is_correct}", path)
        write("--------------------", path)
    else:
        code_line = lines[int(predicted_error_line)-1]
        if code_line != predicted_error_code:
            if predicted_error_code in code_line:
                is_correct = (code_line.replace(" ", "") == actual_error_line.replace(" ", ""))
            else:
                is_correct = (predicted_error_code.replace(" ", "") == actual_error_line.replace(" ", ""))
        else:
            is_correct = (code_line.replace(" ", "") == actual_error_line.replace(" ", ""))
        
        result = {
            "index": idx,
            "predicted": code_line,
            "actual": actual_error_line,
            "is_correct": is_correct
        }
        results.append(result)
        
        write(f"Predicted_1: {code_line}", path)
        write(f"Predicted_2: {predicted_error_code}", path)
        write(f"Actual: {actual_error_line}", path)
        write(f"Is Correct: {is_correct}", path)
        write("--------------------", path)

# Calculate accuracy
accuracy = sum(result["is_correct"] for result in results) / len(results)
write(f"pass@1: {accuracy:.3f}", path)
