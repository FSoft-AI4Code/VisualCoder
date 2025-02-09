from __future__ import annotations
from datasets import load_dataset
from tqdm import tqdm
from src.utils.utils import evaluate_generations, codegen_metrics, write
from src.tasks.debug.lcb_debug import CodeGenerationProblem
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
dataset = load_dataset("huypn16/LCB-R-F")["test"]
 
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
# model = AutoModel.from_pretrained(path, trust_remote_code=True).eval().cuda()
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
pattern = r"```python(.*?)```"  
def get_buggy_cfg_COT_prompt(problem, buggy_program, error_message):
    prompt = f"You are an expert Python programmer. You will be given a question (problem specification), a buggy program and its control flow graph as image and error message, you will generate a correct Python program that matches the specification, fix the original program and passes all the tests. Your task is firstly analayse given information to understand the logic of the code and then think step-by-step logically the way to fix the program before returning final the program. The program should only include function definition with parameter list in order."
    prompt += f"### Question:\n{problem}\n\n"
    prompt += f"### Buggy program:\n```python\n{buggy_program}\n```\n\n"
    prompt += f"### Error message:\n{error_message}\n\n"
    prompt += "```python\n# YOUR CODE HERE\n```"
    return prompt

def get_buggy_COT_prompt(problem, buggy_program, error_message):
    prompt = f"You are an expert Python programmer. You will be given a question (problem specification), a buggy program and its error message, you will generate a correct Python program that matches the specification, fix the original program and passes all the tests. You should think step-by-step logically the way to fix the buggy program before returning the final program. The program should only include function definition with parameter list in order."
    prompt += f"### Question:\n{problem}\n\n"
    prompt += f"### Buggy program:\n```python\n{buggy_program}\n```\n\n"
    prompt += f"### Error message:\n{error_message}\n\n"
    prompt += "```python\n# YOUR CODE HERE\n```"
    return prompt

def get_buggy_prompt(problem, buggy_program, error_message):
    prompt = f"You are an expert Python programmer. You will be given a question (problem specification), a buggy program and its error message, you will generate a correct Python program that matches the specification, fix the original program and passes all the tests. Give me the correct program only (no need any explaination and reasoning)."
    prompt += f"### Question:\n{problem}\n\n"
    prompt += f"### Buggy program:\n```python\n{buggy_program}\n```\n\n"
    prompt += f"### Error message:\n{error_message}\n\n"
    prompt += "```python\n# YOUR CODE HERE\n```"
    return prompt

def get_buggy_cfg_prompt(problem, buggy_program, error_message):
    prompt = f"You are an expert Python programmer. You will be given a question (problem specification), a buggy program and its control flow graph as image and error message, you will generate a correct Python program that matches the specification, fix the original program and passes all the tests. Give me the correct program only (no need any explaination and reasoning)."
    prompt += f"### Question:\n{problem}\n\n"
    prompt += f"### Buggy program:\n```python\n{buggy_program}\n```\n\n"
    prompt += f"### Error message:\n{error_message}\n\n"
    prompt += "```python\n# YOUR CODE HERE\n```"
    return prompt

def get_next(question: CodeGenerationProblem, buggy_program: str, error_message: str, trace: str):
    prompt = f"You are an expert Python programmer. You will be given a question (problem specification), a buggy program with in-line execution traces and its error message, you will generate a correct Python program that matches the specification, fix the original program and passes all the tests. Give me the correct program only (no need any explaination and reasoning)."
    prompt += f"### Question:\n{question}\n\n"
    prompt += f"### Buggy program:\n```python\n{buggy_program}\n```\n\n"
    prompt += f"### Error message:\n{error_message}\n\n"
    prompt += f"### Traces:\n```python\n{trace}\n```\n\n"
    prompt += "```python\n# YOUR CODE HERE\n```"
    return prompt

def get_VisualCoder_prompt(question: CodeGenerationProblem, buggy_program: str, error_message: str):
    prompt = f"You are an expert Python programmer. You will be given a question (problem specification), a buggy program and its control flow graph as image and error message, you will generate a correct Python program that matches the specification, fix the original program and passes all the tests. Your task is firstly analayse given code and refer to corresponsing node in CFG to understand the logic of the code and then think step-by-step logically the way to fix the program before returning final the program. The program should only include function definition with parameter list in order."
    prompt += f"### Question:\n{question}\n\n"
    prompt += f"### Buggy program:\n```python\n{buggy_program}\n```\n\n"
    prompt += f"### Error message:\n{error_message}\n\n"
    prompt += "```python\n# YOUR CODE HERE\n```"
    return prompt

def get_rationale_generation_prompt(problem, buggy_program, error_message):
    prompt = f"""You are an expert Python programmer.

You will be provided with a question (problem specification), a buggy Python program, and an error message. Please analyze the buggy program and generate a detailed rationale explaining why the program fails.

### Question:
{problem}

### Buggy program:
```python
{buggy_program}
```

### Error message:
{error_message}

Please think through the code, step by step, to explain what is wrong with the program before fixing it.""" 
    return prompt

def get_rationale_generation_VisualCoder_prompt(problem, buggy_program, error_message):
    prompt = f"""You are an expert Python programmer.

You will be provided with a question (problem specification), a buggy Python program, and an error message. Please analyze the buggy program and generate a detailed rationale explaining why the program fails.

### Question:
{problem}

### Buggy program:s
```python
{buggy_program}
```

### Error message:
{error_message}

Please think through the code, step by step, refer to the CFG to identify which node corresponds to the line you're currently analyzing to explain what is wrong with the program (only need rationale, no need code).""" 

def get_answer_inference_prompt(problem, buggy_program, error_message, rationale):
    prompt = f"""You are an expert Python programmer.

You have been provided with a buggy Python program, its error message, and the following rationale for why the program fails:

### Rationale:
{rationale}

### Buggy program:
```python
{buggy_program}

### Error message:
{error_message}

Based on this rationale, please fix the program so that it satisfies the problem's requirements.
```python
# YOUR CODE HERE
```"""
    return prompt


def run_code_repair(problem, buggy_program, error_message, filepath, generation_config, trace):
    if 'cfg' in args.setting:
        pixel_values = load_image(filepath, max_num=12).to(torch.bfloat16).cuda()
    else:
        pixel_values = None
    if args.setting == "buggy_COT":
        prompt = get_buggy_COT_prompt(problem, buggy_program, error_message)
 
    elif args.setting == "buggy_cfg_COT":
        prompt = get_buggy_cfg_COT_prompt(problem, buggy_program, error_message)

    elif args.setting == "buggy":
        prompt = get_buggy_prompt(problem, buggy_program, error_message)

    elif args.setting == "buggy_cfg":
        prompt = get_buggy_cfg_prompt(problem, buggy_program, error_message)

    elif args.setting == "VisualCoder":
        prompt = get_VisualCoder_prompt(problem, buggy_program, error_message)

    elif args.setting == "NeXT":
        prompt = get_next(problem, buggy_program, error_message, trace)

    try:
        solution, _ = model.chat(tokenizer, pixel_values, prompt, generation_config, history=None, return_history=True)
    except:
        solution = ""
    # Extract Python code from the solution
    match = re.search(pattern, solution, re.DOTALL)
    if match:
        python_code = match.group(1).strip()
    else:
        print("Cannot find python code in the solution")
        python_code = ""
    python_code = re.sub(r"assert.*|print.*", "", python_code)
    return solution, python_code

def run_code_repair_two_stage(problem, buggy_program, error_message, filepath, generation_config):
    # Step 1: Rationale Generation (Stage 1)
    if args.setting == "Multimodal-CoT":
        rationale_prompt = get_rationale_generation_prompt(problem, buggy_program, error_message)
    elif args.setting == "Multimodal-CoT_VisualCoder":
        rationale_prompt = get_rationale_generation_VisualCoder_prompt(problem, buggy_program, error_message)

    if 'cfg' in args.setting:
        pixel_values = load_image(filepath, max_num=12).to(torch.bfloat16).cuda()
    else:
        pixel_values = None

    # Generate rationale using the model
    try:
        rationale, _ = model.chat(tokenizer, pixel_values, rationale_prompt, generation_config, history=None, return_history=True)
    except:
        rationale = ""
    rationale = rationale.replace("```python", "").replace("```", "")
    
    # Step 2: Code Repair (Stage 2) using the generated rationale
    repair_prompt = get_answer_inference_prompt(problem, buggy_program, error_message, rationale)

    try:
        solution, _ = model.chat(tokenizer, pixel_values, repair_prompt, generation_config, history=None, return_history=True)
    except:
        solution = ""
    # Extract Python code from the solution
    match = re.search(pattern, solution, re.DOTALL)
    if match:
        python_code = match.group(1).strip()
    else:
        print("Cannot find python code in the solution")
        python_code = ""
    
    # Clean up unnecessary lines
    python_code = re.sub(r"assert.*|print.*", "", python_code)
    return rationale, solution, python_code


results = []
path = "output/code_repair/InternVL"
if not os.path.exists(path):
    os.makedirs(path)
path = os.path.join(path, f"{args.setting}_{args.session}.txt")
generations_all = []
samples = []
responses_all = []
solutions_all = []
for idx in tqdm(range(33,len(dataset))):
    problem = CodeGenerationProblem(dataset[idx])
    generations = []
    solutions = []
    sample = problem.get_evaluation_sample()
    write(f"Problem {idx}", path)
    
    for j, buggy_program in enumerate(dataset[idx]["testable_programs"]):
        buggy_program = re.sub(r"assert.*|print.*", "", buggy_program)
        error_message = dataset[idx]["error_messages"][j]
        buggy_program = '\n'.join([line for line in buggy_program.split('\n') if not line.strip().startswith('#')])
        buggy_program = '\n'.join([line for line in buggy_program.split('\n') if not line.strip().startswith('@')])
        buggy_program = re.sub(r'(?<!["\'=#])\s*#.*$', '', buggy_program, flags=re.MULTILINE)
        if 'cfg' in args.setting or args.setting == "Multimodal-CoT" or args.setting == "Multimodal-CoT_VisualCoder":
            filepath, _ = code2cfgimage(buggy_program, filename=f"{args.setting}_{args.session}")
        else:
            filepath = None
        generation_config = dict(max_new_tokens=1024, do_sample=False)
        write(f"Generation {j}", path)
        if args.setting == "Multimodal-CoT" or args.setting == "Multimodal-CoT_VisualCoder":
            rationale, solution, python_code = run_code_repair_two_stage(problem.question_content, buggy_program, error_message, filepath, generation_config)
            write(f"Rationale: {rationale}", path)
            write(f"Solution:", path)
        else:
            solution, python_code = run_code_repair(problem.question_content, buggy_program, error_message, filepath, generation_config)
        generations.append(python_code)
        
        solutions.append(solution)
        write(f"{solution}", path)
        write("____________________", path)
    
    solutions_all.append(solutions)
    generations_all.append(generations)
    samples.append(sample)

results, metadata = evaluate_generations(samples, generations_all, num_process_evaluate=32, timeout=6)
metric = codegen_metrics(samples, generations_all)
write(f'{results}', path)
write(f'{metadata}', path)
write(f'{metric}', path)
print(metric[0])