from __future__ import annotations
from src.utils.utils import write
from src.codetransform.cfg2image import code2cfgimage
import torch.nn.functional as F
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
import os
import re
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import gc
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
    image = image.resize((448,448))
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
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)
pattern = r"```python(.*?)```"  

def get_buggy_cfg_COT_oneshot_prompt(code: str):
    prompt = f"""Analyze step-by-step the following Python code snippet, which contains error(s) when executing

You will also be provided with a control flow graph (CFG) image of this code. As you analyze each line:
1. Examine each line of code sequentially.
2. Use this understanding to support your reasoning about the code's logic and potential errors.

Think through your analysis step by step, considering both the code and its representation in the CFG.

Here is an example:
Code:
n = '5'
A = [2, 1, 5, 4, 3]
ans = 0 
temp = 0
for i in range(n-1):
    if A[i]<temp:
        ans += temp-A[i]
    else:
        temp = max(temp,A[i])
print(ans)

Analysis:
The code is intended to calculate the answer to a problem involving an array `A` and a variable `temp`. The logic is as follows:

1. Initialize `n` to '5', `A` to `[2, 1, 5, 4, 3]`, `ans` to 0, and `temp` to 0.
2. Loop through the array `A` from the first element to the second last element.
3. If the current element `A[i]` is less than `temp`, add `temp - A[i]` to `ans`.
4. Otherwise, update `temp` to the maximum of its current value and `A[i]`.
5. Print the final value of `ans`.

However, there is a problem with the code:

- The variable `n` is a string, not an integer. This will cause issues when used in the `range()` function, as Python cannot subtract one string from another.

To fix this, `n` should be converted to an integer before using it in the `range()` function.

The problematic line of code is:

```python
for i in range(n-1):
```
Code: {code}
"""
    return prompt

def get_buggy_cfg_COT_zeroshot_prompt(code: str):
    prompt = f"""Analyze the following Python code snippet, which contains error(s) when executing:

{code}

You will also be provided with a control flow graph (CFG) image of this code. As you analyze each line:
1. Examine each line of code sequentially.
2. Use this understanding to support your reasoning about the code's logic and potential errors.

Think through your analysis step by step.

After your analysis, respond with only the line number and the problematic line of code that causes termination, separated by a colon. For example: '3: problematic_line_of_code_here'.
"""
    return prompt

def get_VisualCoder_oneshot_prompt(code: str):
    prompt = f"""Analyze step-by-step the following Python code snippet, which contains error(s) when executing

You will also be provided with a control flow graph (CFG) image of this code. As you analyze each line:
1. Examine each line of code sequentially
2. Refer to the CFG to identify which node corresponds to the line you're currently analyzing for example: "This line corresponds to the node at the top of the CFG".
3. Use this alignment between code and CFG to support your reasoning about the code's logic and potential errors.

Think through your analysis step by step, considering both the code and its representation in the CFG.
Here is an example:
Code:
N=16
if N[-1] == 2 or 4 or 5 or 7 or 9:
    print("hon")
if N[-1] == 0 or 1 or 6 or 8:
    print("pon")
if N[-1] == 3:
    print("bon")
if N[-1] == 2 or N[-1] == 4 or N[-1] == 5 or N[-1] == 7 or N[-1] == 9:
    print("hon")
if N[-1] == 0 or N[-1] == 1 or N[-1] == 6 or N[-1] == 8:
    print("pon")

Analysis:
Let's analyze the code step by step and compare it with the control flow graph (CFG) to identify the problematic line.

1. **Initialization:**
   ```python
   N = 16
   ```
   - This line initializes the variable `N` to 16. This corresponds to the node at the top of the CFG.

2. **First `if` statement:**
   ```python
   if N[-1] == 2 or 4 or 5 or 7 or 9:
       print("hon")
   ```
   - This line checks if the last element of `N` (which is 6 since Python uses 0-based indexing) is equal to 2, 4, 5, 7, or 9. This corresponds to the first `if` node in the CFG.

3. **Second `if` statement:**
   ```python
   if N[-1] == 0 or 1 or 6 or 8:
       print("pon")
   ```
   - This line checks if the last element of `N` is equal to 0, 1, 6, or 8. This corresponds to the second `if` node in the CFG.

4. **Third `if` statement:**
   ```python
   if N[-1] == 3:
       print("bon")
   ```
   - This line checks if the last element of `N` is equal to 3. This corresponds to the third `if` node in the CFG.

Now, let's identify the problematic line:

- The first two `if` statements use the `or` operator without parentheses, which can lead to unexpected behavior due to operator precedence. The correct way to write these conditions would be:
  ```python
  if N[-1] == 2 or N[-1] == 4 or N[-1] == 5 or N[-1] == 7 or N[-1] == 9:
      print("hon")
  if N[-1] == 0 or N[-1] == 1 or N[-1] == 6 or N[-1] == 8:
      print("pon")
  ```

- The code as written will evaluate the conditions from left to right and stop at the first true condition, without evaluating the rest. This can lead to incorrect results if the intention was to check all conditions.

Therefore, the problematic line is:

**3: if N[-1] == 2 or 4 or 5 or 7 or 9:**

Code: {code}
Analysis:"""
    return prompt

def get_VisualCoder_zeroshot_prompt(code: str):
    prompt = f"""Analyze the following Python code snippet, which contains error(s) when executing:

{code}

You will also be provided with a control flow graph (CFG) image of this code. As you analyze each line:
1. Examine each line of code sequentially.
2. Reference the CFG to identify which node corresponds to the line you're currently analyzing.
3. Use this alignment between code and it's CFG image to support your reasoning about the code's logic and potential errors.

Think through your analysis step by step, considering both the code and its representation in the CFG image.

After your analysis, respond with only the line number and the problematic line of code that causes termination, separated by a colon. For example: '3: problematic_line_of_code_here'.
"""
    return prompt
def get_cfg_COT_prompt(code: str):
    prompt = f"""Analyze the following control flow graph (CFG) of a Python code which contains error(s) when executing.As you analyze each line:

1. Examine each line of code in the provided control flow graph (CFG) image.
2. Use this understanding to support your reasoning about the code's logic and potential errors.

Think through your analysis step by step.

After your analysis, respond with only the line number and the problematic line of code that causes termination, separated by a colon. For example: '3: problematic_line_of_code_here'.
"""
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
    elif args.setting == "buggy_cfg_COT" and args.prompt_mode == "oneshot":
        prompt = get_buggy_cfg_COT_oneshot_prompt(code)
        response, generation_output, query = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=False, analysis=analysis)
    elif args.setting == "buggy_cfg_COT" and args.prompt_mode == "zeroshot":
        prompt = get_buggy_cfg_COT_zeroshot_prompt(code)
        response, generation_output, query = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=False, analysis=analysis)
    elif args.setting == "cfg_COT":
        prompt = get_cfg_COT_prompt(code)
        response, generation_output, query = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=False, analysis=analysis)
    elif args.setting == "buggy":
        prompt = get_buggy_prompt(code)
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
    elif args.setting == "cfg":
        prompt = get_cfg_prompt(code)
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
    elif args.setting == "buggy_cfg": 
        prompt = get_buggy_cfg_prompt(code)
        response, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
    elif args.setting == "VisualCoder" and args.prompt_mode == "oneshot":
        prompt = get_VisualCoder_oneshot_prompt(code)
        response, generation_output, query = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=False, analysis=analysis)
    elif args.setting == "VisualCoder" and args.prompt_mode == "zeroshot":
        prompt = get_VisualCoder_zeroshot_prompt(code)
        response, generation_output, query = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=False, analysis=analysis)
    return response, generation_output, query

def parse_steps(response_text, buggy_mode=False):
    num_step = 1
    content = []
    find_content = False
    steps = []
    for line in response_text.split("\n"):
        if "." in line:
            num = line.split(".")[0]
        else:
            num = line.split(":")[0]
        if num.isdigit():
            if int(num) == num_step:
                if find_content:
                    content_text = "\n".join(content)
                    start_char = response_text.find(content_text)
                    end_char = start_char + len(content_text)
                    steps.append({
                        'num': num_step-1,
                        'title': f"Step {num_step-1}",
                        'start_char': start_char,
                        'end_char': end_char,
                        'content': content_text
                    })
                    content = []
                    content.append(line)
                    num_step += 1
                else:
                    find_content = True
                    content.append(line)
                    num_step += 1
            else:
                if find_content:
                    content.append(line)
        else:
            if find_content:
                if line:
                    content.append(line)
                else:
                    content_text = "\n".join(content)
                    start_char = response_text.find(content_text)
                    end_char = start_char + len(content_text)
                    steps.append({
                        'num': num_step-1,
                        'title': f"Step {num_step-1}",
                        'start_char': start_char,
                        'end_char': end_char,
                        'content': content_text
                    })
                    content = []
                    find_content = False
    return steps

def find_subsequence(full_tokens, step_tokens, start_index=0):
    """
    Finds the start index of step_tokens within full_tokens starting from start_index.
    Returns the start index if found, else -1.
    """
    len_full = len(full_tokens)
    len_step = len(step_tokens)
    
    for i in range(start_index, len_full - len_step + 1):
        if full_tokens[i:i+len_step] == step_tokens:
            return i
    return -1

def assign_tokens_sequential(full_text, steps, tokenizer):
    """
    Assigns token indices to each step by searching for step token sequences within full tokens.
    """
    # Tokenize the full text
    full_tokens = tokenizer.encode(full_text, add_special_tokens=False)
    current_index = 0
    # import ipdb; ipdb.set_trace()
    for step in steps:
        # Tokenize the step content
        step_tokens = tokenizer.encode(step['content'], add_special_tokens=False)
        # Search for the step token sequence in the full token list
        start_idx = find_subsequence(full_tokens, step_tokens[:len(step_tokens)//2+1], start_index=current_index)
        if start_idx == -1:
            print(f"Token sequence for Step {step['num']}: '{step['title']}' not found.")
            step['tokens'] = []
            continue
        # Assign tokens to the step
        step['tokens'] = list(range(start_idx, start_idx + len(step_tokens)))
        # Update the current_index to the end of this step's tokens
        current_index = start_idx + len(step_tokens)
        print(f"Assigned tokens for Step {step['num']}: '{step['title']}'")
    return steps

def compute_average_attention_per_step(steps, llm_attn_matrix, vision_token_start, len_prompt_tokens, vision_token_end):
    """
    Computes the average attention over vision tokens for each step.
    """
    average_attn_per_step = []
    step_labels = []
    for step in steps:
        if not step['tokens']:
            continue  # Skip steps with no tokens
        # Extract the attention rows corresponding to the step's tokens
        step_token = step['tokens']
        step_token = [token + len_prompt_tokens for token in step_token]
        step_attn = llm_attn_matrix[step_token, vision_token_start:vision_token_end]
        # Sum attention over vision tokens for each token in the step
        step_attn_sum = step_attn.sum(dim=1)  # Shape: [num_tokens_in_step]
        # Compute the average attention for the step
        step_attn_avg = step_attn_sum.mean().item()
        average_attn_per_step.append(step_attn_avg)
        step_labels.append(step['title'])
    return average_attn_per_step, step_labels

def plot_average_attention(average_attn, labels, output_path="step_token_attention.png"):
    """
    Plots the average attention per step with step titles as x-axis labels.
    """
    fig, ax = plt.subplots(figsize=(30, 5))
    ax.plot(average_attn, marker='o')
    ax.set_xticks(range(len(average_attn)))
    ax.set_xticklabels(labels, rotation=75, ha='right')
    ax.set_xlabel("Steps")
    ax.set_ylabel("Average Attention over Vision Tokens")
    ax.set_title("Average Attention per Step")
    ax.grid(True)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Plot saved to {output_path}")


def plot_steps_attention_heatmaps(
    steps,
    llm_attn_matrix,
    vision_token_start,
    vision_token_end,
    image,
    model,
    tokenizer,
    len_prompt_tokens,
    output_dir="steps_cot_only_attention",
):
    """
    Plots and saves attention heatmaps for each step.

    Args:
        steps (list of dict): List of steps, each containing 'num', 'title', and 'tokens' keys.
        llm_attn_matrix (torch.Tensor or np.ndarray): Attention matrix of shape [seq_len, num_vision_tokens].
        vision_token_start (int): Start index of vision tokens in the attention matrix.
        vision_token_end (int): End index of vision tokens in the attention matrix.
        image (PIL.Image or np.ndarray): The original image.
        model: The model object containing vision attentions.
        tokenizer: The tokenizer used for decoding tokens.
        output_dir (str): Directory to save the heatmap images.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Aggregate vision attention
    # vis_attn_matrix = aggregate_vit_attention(
    #     model.vision_model.image_attentions,
    #     select_layer=model.select_layer,
    #     all_prev_layers=True
    # )
    grid_size = model.vision_model.embeddings.num_patches_per_side

    # Convert attention matrices to numpy if they are torch tensors
    if isinstance(llm_attn_matrix, torch.Tensor):
        llm_attn_matrix = llm_attn_matrix.cpu().numpy()
    # if isinstance(vis_attn_matrix, torch.Tensor):
    #     vis_attn_matrix = vis_attn_matrix.float().cpu().numpy()

    for step in steps:
        step_num = step['num']
        step_title = step['title']
        token_indices = step.get('tokens', [])
        text_content = step.get('content', '')
        # add token_indices by len_prompt_tokens
        token_indices = [token_index + len_prompt_tokens for token_index in token_indices]
        
        if not token_indices:
            print(f"Skipping Step {step_num}: '{step_title}' due to no assigned tokens.")
            continue
        
        # Aggregate attention weights for all tokens in the step
        # Shape of llm_attn_matrix[token_indices, vision_token_start:vision_token_end]: [num_tokens, num_vision_tokens]
        step_attn = llm_attn_matrix[token_indices, vision_token_start:vision_token_end]
        
        # Option 1: Sum attention over all tokens in the step
        # import ipdb; ipdb.set_trace()
        aggregated_attn = step_attn.sum(axis=0)
        
        # Option 2: Average attention over all tokens in the step
        # aggregated_attn = step_attn.mean(axis=0)
        
        # Normalize the aggregated attention
        # if aggregated_attn.sum() != 0:
        #     aggregated_attn /= aggregated_attn.sum()
        
        # Reshape attention to image grid
        # attn_over_image = []
        # for weight, vis_attn in zip(aggregated_attn, vis_attn_matrix):
        #     # import ipdb; ipdb.set_trace()
        #     vis_attn = vis_attn.reshape(grid_size, grid_size)
        #     attn_over_image.append(vis_attn * weight)
        # attn_over_image = 
        try:
            attn_over_image = np.array(aggregated_attn).reshape(-1,16,16)
        except:
            try:
                attn_over_image = np.array(aggregated_attn).reshape(1,16,16)
            except:
                attn_over_image = np.array(aggregated_attn).reshape(7,16,16)
        attn_over_image = attn_over_image.mean(axis=0)
        # import ipdb; ipdb.set_trace()
        
        # Normalize attention over the image
        if attn_over_image.max() != 0:
            attn_over_image /= attn_over_image.max()
        # import ipdb; ipdb.set_trace()
        
        # corner_size = attn_over_image.shape[0] // 8
        # attn_over_image[:corner_size, :corner_size] = 0

        # # Shift attention to the left
        # shift_amount = attn_over_image.shape[1] // 8
        # attn_shifted = np.zeros_like(attn_over_image)
        # attn_shifted[:, :-shift_amount] = attn_over_image[:, shift_amount:]
        # attn_over_image = (attn_shifted - attn_shifted.min()) / (attn_shifted.max() - attn_shifted.min())

        # Resize attention to match image size
        attn_over_image = F.interpolate(
            torch.tensor(attn_over_image).unsqueeze(0).unsqueeze(0), 
            size=(image.size[1], image.size[0]),  # (height, width)
            mode='bilinear', 
            # mode='linear',
            align_corners=False
        ).squeeze().numpy()
        
        # Convert image to numpy if it's a PIL image
        if isinstance(image, torch.Tensor):
            np_img = image.permute(1, 2, 0).cpu().numpy()[:, :, ::-1]
        elif isinstance(image, np.ndarray):
            np_img = image[:, :, ::-1]
        else:
            np_img = np.array(image)[:, :, ::-1]
        
        # Overlay attention on image
        img_with_attn, heatmap = show_mask_on_image(np_img, attn_over_image)
        # Remove high scores from top-left corner

                # Plotting (rest of the code remains the same)
        fig, ax = plt.subplots(figsize=(8, 8))
        # Plot the image
        ax.imshow(img_with_attn)
        ax.axis("off")
        
        # # Add text content below the image
        # ax.text(0.01, -0.02, text_content, 
        #         transform=ax.transAxes,
        #         horizontalalignment='left', 
        #         verticalalignment='top',
        #         wrap=True, 
        #         fontsize=10)

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        
        save_path = os.path.join(output_dir, f"{step_title.replace(' ', '_')}.png")
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        print(f"Saved attention heatmap for Step {step_num}: '{step_title}' to {save_path}")

if __name__ == "__main__":
    from src.utils.utils import (
        aggregate_llm_attention, aggregate_vit_attention,
        heterogenous_stack,
        show_mask_on_image
    )
    
    args = src.utils.options.options().parse_args()

    with open("dataset.json", "r") as f:
        dataset = json.load(f)
    results = []
    path = f"output/attention/{args.setting}"
    if not os.path.exists(path):
        os.makedirs(path)
    code = """X = 10
Y = X**2
for i in range(1,Y):
    if i%2 == 0:
        lst = []
    else:
        lst.append(i)
print(lst[0])"""

    if 'cfg' in args.setting or args.setting == "VisualCoder":
        filepath, _ = code2cfgimage(code, filename=f"{args.setting}_{args.session}_InterVL")
    else:
        filepath = None

    # args.setting = "VisualCoder"
    # path = f"output/bug_localization/{args.setting}_{args.session}"
    if not os.path.exists(path):
        os.makedirs(path)
    generation_config = dict(max_new_tokens=2048, do_sample=False, output_attentions=True, return_dict_in_generate=True)
    response, generation_output, query_prompt = run_error_detection(code, filepath, generation_config, analysis=True)
    # try:
    #     response, generation_output, query_prompt = run_error_detection(code, filepath, generation_config, analysis=True)
    # except:
    #     pass
    idx = len(os.listdir(path))
    save_path = os.path.join(path, f"problem_{idx}")
    text_path = os.path.join(save_path, "Answer.txt")
    os.makedirs(save_path, exist_ok=True)
    write(response, text_path)
    write("_____________________", text_path)
    write("Buggy Code", text_path)
    write(code, text_path)
    # write(f"Correct answer: {actual_error_line}", text_path)

    # Aggregate prompt attention
    aggregated_prompt_attention = []
    for i, layer in enumerate(generation_output["attentions"][0]):
        layer_attns = layer.squeeze(0)  # Shape: [num_heads, seq_len]
        attns_per_head = layer_attns.mean(dim=0)  # Shape: [seq_len]
        cur = attns_per_head[:-1].cpu().clone()

        cur[1:, 0] = 0.
        cur[1:] = cur[1:] / cur[1:].sum(-1, keepdim=True)
        aggregated_prompt_attention.append(cur)
    aggregated_prompt_attention = torch.stack(aggregated_prompt_attention).mean(dim=0)  # Shape: [seq_len]

    # Stack LLM attention
    llm_attn_matrix = heterogenous_stack(
        [torch.tensor([1])]
        + list(aggregated_prompt_attention) 
        + list(map(aggregate_llm_attention, generation_output["attentions"]))
    )
    gamma_factor = 1
    enhanced_attn_m = np.power(llm_attn_matrix.numpy(), 1 / gamma_factor)

    # Plot attention matrix
    fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
    ax.imshow(enhanced_attn_m, 
            vmin=enhanced_attn_m.min(), 
            vmax=enhanced_attn_m.max(), 
            interpolation="nearest")
    ax.set_xlabel("Vision Tokens")
    ax.set_ylabel("LLM Tokens")
    ax.set_title("Enhanced Attention Matrix")
    fig.savefig("attention.png")
    plt.close(fig)
    print("Attention matrix plot saved to attention.png")

    # Identify vision tokens
    num_patches = query_prompt.count("<IMG_CONTEXT>")
    input_ids = tokenizer(query_prompt, return_tensors='pt')["input_ids"]
    input_token_len = len(input_ids[0]) - 1  # -1 for the <image> token

    # vision_token_start = len(tokenizer(query_prompt.split("<img>")[0], return_tensors='pt')["input_ids"][0])
    # vision_token_end = vision_token_start + num_patches
    # get all indices of 92546 in tokenizer(query_prompt, return_tensors='pt')["input_ids"][0]
    all_vision_token_indices = (tokenizer(query_prompt, return_tensors='pt')["input_ids"][0] == 92546).nonzero(as_tuple=True)[0]
    if len(all_vision_token_indices) != 256:
        raise ValueError("Number of vision tokens is not 256")
    vision_token_start = all_vision_token_indices[0].item()
    vision_token_end = all_vision_token_indices[-1].item() + 1
    len_prompt_tokens = len(tokenizer(query_prompt, return_tensors='pt')["input_ids"][0])
    # Parse steps from the response
    steps = parse_steps(response, "buggy" in args.setting)
    if not steps:
        print("No steps found in the response. Please ensure the response follows the expected format.")
        exit(1)

    # Assign tokens to steps
    steps = assign_tokens_sequential(response, steps, tokenizer)
    # Compute average attention per step
    average_attn, step_labels = compute_average_attention_per_step(
        steps, llm_attn_matrix, vision_token_start, len_prompt_tokens, vision_token_end
    )

    if not average_attn:
        print("No attention data computed for steps.")
        exit(1)

    # Plot the average attention per step
    plot_average_attention(average_attn, step_labels)

    # Optionally, save step information to a text file
    with open("steps_attention.txt", "w") as f:
        for label, attn in zip(step_labels, average_attn):
            f.write(f"{label}: {attn}\n")
    print("Step attention details saved to steps_attention.txt")

    image = Image.open(filepath).convert("RGB")
    image = image.resize((448, 448))
    plot_steps_attention_heatmaps(steps, llm_attn_matrix, vision_token_start, vision_token_end, image, model, tokenizer, len_prompt_tokens, save_path)

    del response, generation_output, query_prompt, steps, average_attn, step_labels, image, llm_attn_matrix, enhanced_attn_m, input_ids, input_token_len, vision_token_start, vision_token_end, aggregated_prompt_attention, gamma_factor, fig, ax
    gc.collect()
    torch.cuda.empty_cache()