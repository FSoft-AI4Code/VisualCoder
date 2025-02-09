from __future__ import annotations
import json
from tqdm import tqdm
from src.utils.utils import write
from src.codetransform.cfg2image import code2cfgimage
 
import re
import os
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import src.utils.options

args = src.utils.options.options().parse_args()
 
# Load dataset
with open("dataset.json", "r") as f:
    dataset = json.load(f)
 
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
 
def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform
 
def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio
 
def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
 
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
 
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
 
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
 
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images
 
def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values
 
path = 'OpenGVLab/InternVL2-26B'
model = AutoModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
pattern = r"```python(.*?)```"  

def get_buggy_cfg_COT_prompt(code: str):
    prompt = f"""Analyze the following Python code snippet, which contains error(s) when executing:

{code}

You will also be provided with a control flow graph (CFG) image of this code. As you analyze each line:
1. Examine each line of code sequentially using both code and CFG 
2. Use this understanding to support your reasoning about the code's logic and error line that causes termination.

Think through your analysis step by step, considering both the code and its representation in the CFG.

After your analysis, respond with only the line number and the problematic line of code that causes termination, separated by a colon. For example: '3: problematic_line_of_code_here'."""
    return prompt

def get_VisualCoder_prompt(code: str):
    prompt = f"""Analyze the following Python code snippet, which contains error(s) when executing:

{code}

You will also be provided with a control flow graph (CFG) image of this code. As you analyze each line:
1. Examine each line of code in plain code sequentially.
2. Reference the CFG to identify which node corresponds to the line you're currently analyzing.
3. Use this alignment between code and CFG to support your reasoning about the code's logic and error line that causes termination.

Think through your analysis step by step, considering both the code and its representation in the CFG.

After your analysis, respond with only the line number and the problematic line of code that causes termination, separated by a colon. For example: '3: problematic_line_of_code_here'."""
    return prompt


def get_rationale_generation_VisualCoder_prompt(code: str):
    prompt = f"""Analyze the following Python code snippet, which contains error(s) when executing:

{code}

You will also be provided with a control flow graph (CFG) image of this code if applicable. As you analyze each line:
1. Examine each line of code in plain code sequentially.
2. Reference the CFG to identify which node corresponds to the line you're currently analyzing.
3. Use this alignment between code and CFG to support your reasoning about the code's logic and potential errors.

After your analysis, provide a detailed rationale explaining what might be wrong with the code."""

    return prompt

def get_rationale_generation_prompt(code: str):
    prompt = f"""Analyze the following Python code snippet, which contains error(s) when executing:

{code}

You will also be provided with a control flow graph (CFG) image of this code if applicable. As you analyze each line:
1. Examine each line of code sequentially using both code and CFG 
2. Use this understanding to generate a rationale for where the code might terminate.

After your analysis, provide a detailed rationale explaining what might be wrong with the code."""

    return prompt

def get_answer_inference_prompt(code: str, rationale: str):
    prompt = f"""You have a Python code snippet containing error(s) and a rationale for the error(s).

Code:

{code}

Rationale: {rationale}

Using this rationale, please identify the specific line of code that causes termination. Respond with only the line number and the problematic line of code that causes termination, separated by a colon. For example: '3: problematic_line_of_code_here'."""

    return prompt


def get_buggy_COT_prompt(code: str):
    prompt = f"""Analyze the following Python code snippet, which contains error(s) when executing:

{code}

You will also be provided with a control flow graph (CFG) image of this code if applicable. As you analyze each line:
1. Examine each line of code sequentially using both code and CFG 
2. Use this understanding to generate a rationale for where the code might terminate.

After your analysis, respond with only the line number and the problematic line of code that causes termination, separated by a colon. For example: '3: problematic_line_of_code_here'."""

    return prompt

def get_buggy_prompt(code: str):
    prompt = f"""You are an expert Python programmer. You will be given a Python code snippet containing error(s). \n```python\n{code}\n```\nYour task is identify the specific line in the code that make the code terminate. Please respond with only the line number and the problematic line of code, separated by a colon. For example: '3: problematic_line_of_code_here'"""
    return prompt

def get_buggy_cfg_prompt(code: str):
    prompt = f"""You are an expert Python programmer. You will be given a Python code snippet and its control flow graph image. The code contains error(s). \n```python\n{code}\n```\n Your task is identify the specific line in the code that make the code terminate. Please respond with only the line number and the problematic line of code, separated by a colon. For example: '3: problematic_line_of_code_here'"""
    return prompt




def run_error_detection(code, image_path, generation_config):
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    if args.setting == "buggy_COT":
        prompt = get_buggy_COT_prompt(code)
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
    elif args.setting == "buggy_cfg_COT":
        prompt = get_buggy_cfg_COT_prompt(code)
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
    elif args.setting == "buggy":
        prompt = get_buggy_prompt(code)
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
    elif args.setting == "buggy_cfg": 
        prompt = get_buggy_cfg_prompt(code)
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
    elif args.setting == "VisualCoder":
        prompt = get_VisualCoder_prompt(code)
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
    return response

def run_error_detection_multimodal_cot(code, image_path, generation_config):
    pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    
    if args.setting == "Multimodal-CoT":
        prompt_stage1 = get_rationale_generation_prompt(code)
    elif args.setting == "Multimodal-CoT_VisualCoder":
        prompt_stage1 = get_rationale_generation_VisualCoder_prompt(code)
    response_rationale, _ = model.chat(tokenizer, pixel_values, prompt_stage1, generation_config, history=None, return_history=True)
    # Stage 2: Answer Inference
    prompt_stage2 = get_answer_inference_prompt(code, response_rationale)
    response_answer, _ = model.chat(tokenizer, None, prompt_stage2, generation_config, history=None, return_history=True)

    return response_rationale, response_answer


results = []
path = "output/bug_localization/InternVL"
if not os.path.exists(path):
    os.makedirs(path)
path = os.path.join(path, f"{args.setting}_{args.session}.txt")

for idx in tqdm(range(len(dataset))):
    data_point = dataset[idx]
    code = data_point["code"]
    actual_error_line = data_point["error_line"]
    if 'cfg' in args.setting or args.setting == "VisualCoder" or args.setting == "Multimodal-CoT" or args.setting == "Multimodal-CoT_VisualCoder":
        filepath, _ = code2cfgimage(code, filename=f"{args.setting}_{args.session}_InterVL")
    else:
        filepath = None

    generation_config = dict(max_new_tokens=2048, do_sample=False)
    write(f"Problem {idx}", path)
    if args.setting == "Multimodal-CoT" or args.setting == "Multimodal-CoT_VisualCoder":
        rationale, response = run_error_detection_multimodal_cot(code, filepath, generation_config)
        write(f"Rationale: {rationale}", path)
        write("-------->", path)
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