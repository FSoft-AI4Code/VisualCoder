from __future__ import annotations
from datasets import load_dataset, load_from_disk
import json
from dataclasses import dataclass
from enum import Enum
import zlib
import pickle
from tqdm import tqdm
from src.utils.utils import evaluate_generations, codegen_metrics, write
from src.tasks.debug.lcb_debug import CodeGenerationProblem
from src.codetransform.cfg2image import code2cfgimage
 
import re
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer
from src.models.internvl2.modeling_internvl_chat import InternVLChatModel
import src.utils.options

args = src.utils.options.options().parse_args()
 
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
model = InternVLChatModel.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    # low_cpu_mem_usage=True,
    attn_implementation="eager",
    # use_flash_attn=True,
    trust_remote_code=True).eval().cuda()
# model = AutoModel.from_pretrained(path, trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
pattern = r"```python(.*?)```"  

def get_buggy_cfg_COT_prompt(code: str):
    prompt = f"""Analyze the following Python code snippet, which contains error(s) when executing:

{code}

You will also be provided with a control flow graph (CFG) image of this code. As you analyze each line:
1. Examine each line of code sequentially using both code and CFG 
2. Use this understanding to support your reasoning about the code's logic and potential errors.

Think through your analysis step by step, considering both the code and its representation in the CFG.

After your analysis, respond with only the line number and the problematic line of code that causes termination, separated by a colon. For example: '3: problematic_line_of_code_here'."""
    return prompt

def get_VisualCoder_prompt(code: str):
    prompt = f"""Analyze the following Python code snippet, which contains error(s) when executing:

{code}

You will also be provided with a control flow graph (CFG) image of this code. As you analyze each line:
1. Primarily focus on the plain code.
2. Reference the CFG to identify which node corresponds to the line you're currently analyzing.
3. Use this alignment between code and CFG to support your reasoning about the code's logic and potential errors.

Think through your analysis step by step, considering both the code and its representation in the CFG.

After your analysis, respond with only the line number and the problematic line of code that causes termination, separated by a colon. For example: '3: problematic_line_of_code_here'."""
    return prompt

def get_cfg_COT_prompt(code: str):
    prompt = f"""You are an expert Python programmer. You will be given a control flow graph image of a Python code containing error(s). Your task is firstly analyse control flow graph to understand the logic of the code and the dependencies between code blocks, then logically think step by step to identify the specific line in the code that make the code terminate. After analysing and reasoning, please respond with only the line number and the problematic line of code, separated by a colon. For example: '3: problematic_line_of_code_here'"""
    return prompt

def get_buggy_COT_prompt(code: str):
    prompt = f"""You are an expert Python programmer. You will be given a Python code snippet containing error(s). \n```python\n{code}\n```\nYour task is firstly analyse code to understand the logic of the code, then logically think step by step to identify the specific line in the code that make the code terminate. After analysing and reasoning, please respond with only the line number and the problematic line of code, separated by a colon. For example: '3: problematic_line_of_code_here'"""
    return prompt

def get_buggy_prompt(code: str):
    prompt = f"""You are an expert Python programmer. You will be given a Python code snippet containing error(s). \n```python\n{code}\n```\nYour task is identify the specific line in the code that make the code terminate. Please respond with only the line number and the problematic line of code, separated by a colon. For example: '3: problematic_line_of_code_here'"""
    return prompt

def get_cfg_prompt(code:str):
    prompt = f"""You are an expert Python programmer. You will be given a control flow graph image of a Python code containing error(s). Your task is identify the specific line in the code that make the code terminate. Please respond with only the line number and the problematic line of code, separated by a colon. For example: '3: problematic_line_of_code_here'"""
    return prompt

def get_buggy_cfg_prompt(code: str):
    prompt = f"""You are an expert Python programmer. You will be given a Python code snippet and its control flow graph image. The code contains error(s). \n```python\n{code}\n```\n Your task is identify the specific line in the code that make the code terminate. Please respond with only the line number and the problematic line of code, separated by a colon. For example: '3: problematic_line_of_code_here'"""
    return prompt


def run_error_detection(code, image_path, generation_config, analysis=False):
    if 'cfg' in args.setting or args.setting == "VisualCoder":
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()
    else:
        pixel_values = None
    if args.setting == "buggy_COT":
        prompt = get_buggy_COT_prompt(code)
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=False)
    elif args.setting == "buggy_cfg_COT":
        prompt = get_buggy_cfg_COT_prompt(code)
        response, generation_output, query = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=False, analysis=analysis)
    elif args.setting == "cfg_COT":
        prompt = get_cfg_COT_prompt(code)
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
    elif args.setting == "buggy":
        prompt = get_buggy_prompt(code)
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
    elif args.setting == "cfg":
        prompt = get_cfg_prompt(code)
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
    elif args.setting == "buggy_cfg": 
        prompt = get_buggy_cfg_prompt(code)
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
    elif args.setting == "VisualCoder":
        prompt = get_VisualCoder_prompt(code)
        response, generation_output, query = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=False, analysis=analysis)
    return response, generation_output, query


if __name__ == "__main__":
    from src.utils.utils import (
        aggregate_llm_attention, aggregate_vit_attention,
        heterogenous_stack,
        show_mask_on_image
    )
    
    results = []
    path = "new_output/bug_localization/InternVL"
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, f"{args.setting}_{args.session}.txt")

    code = """S = ['1', '2', '1', '4']
K = 4
for i in range(len(K)):
    if S[i] != 1:
        print(S[i])
        exit()
print(1)"""

    if 'cfg' in args.setting or args.setting == "VisualCoder":
        filepath, _ = code2cfgimage(code, filename=f"{args.setting}_{args.session}_InterVL")
    else:
        filepath = None

    generation_config = dict(max_new_tokens=2048, do_sample=False, output_attentions=True, return_dict_in_generate=True)
    response, generation_output, query_prompt = run_error_detection(code, filepath, generation_config, analysis=True)
    print("Response: ", response)
    
    aggregated_prompt_attention = []
    for i, layer in enumerate(generation_output["attentions"][0]):
        layer_attns = layer.squeeze(0)
        attns_per_head = layer_attns.mean(dim=0)
        cur = attns_per_head[:-1].cpu().clone()

        cur[1:, 0] = 0.
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        aggregated_prompt_attention.append(cur)
    aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)

    llm_attn_matrix = heterogenous_stack(
        [torch.tensor([1])]
        + list(aggregated_prompt_attention) 
        + list(map(aggregate_llm_attention, generation_output["attentions"]))
    )
    gamma_factor = 1
    enhanced_attn_m = np.power(llm_attn_matrix.numpy(), 1 / gamma_factor)

    fig, ax = plt.subplots(figsize=(10, 20), dpi=150)
    ax.imshow(enhanced_attn_m, vmin=enhanced_attn_m.min(), vmax=enhanced_attn_m.max(), interpolation="nearest")
    fig.savefig("attention.png")
    
    num_patches = query_prompt.count("<IMG_CONTEXT>")
    input_ids = tokenizer(query_prompt, return_tensors='pt')["input_ids"]
    input_token_len = len(input_ids[0]) - 1 # -1 for the <image> token
    
    vision_token_start = len(tokenizer(query_prompt.split("<img>")[0], return_tensors='pt')["input_ids"][0])
    vision_token_end = vision_token_start + num_patches
    overall_attn_weights_over_vis_tokens = []
    
    for i, (row, token) in enumerate(
        zip(
            llm_attn_matrix[input_token_len:], 
            generation_output["sequences"][0].tolist()
        )
    ):
        overall_attn_weights_over_vis_tokens.append(
            row[vision_token_start:vision_token_end].sum().item()
        )

    # plot the trend of attention weights over the vision tokens
    fig, ax = plt.subplots(figsize=(30, 5))
    ax.plot(overall_attn_weights_over_vis_tokens)
    ax.set_xticks(range(len(overall_attn_weights_over_vis_tokens)))
    ax.set_xticklabels(
        [tokenizer.decode(token, add_special_tokens=False).strip() for token in generation_output["sequences"][0].tolist()],
        rotation=75
    )
    ax.set_title("at each token, the sum of attention weights over all the vision tokens")
    fig.savefig("token_attention.png")