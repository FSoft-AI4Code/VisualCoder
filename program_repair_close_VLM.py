from __future__ import annotations
from datasets import load_dataset
from tqdm import tqdm
from src.utils.utils import evaluate_generations, codegen_metrics, write
from src.tasks.debug.lcb_debug import CodeGenerationProblem
from src.codetransform.cfg2image import code2cfgimage
from src.codetransform.next import execute_and_trace
from src.utils.utils import write
import src.utils.options
import requests
import anthropic
import re
import os
import base64

args = src.utils.options.options().parse_args()

dataset = load_dataset("huypn16/LCB-R-F")["test"]

pattern = r"```python(.*?)```"  
SYSTEM_MESSAGE_GENERIC = f"You are an expert Python programmer."

args = src.utils.options.options().parse_args()

def get_buggy_cfg_COT_prompt(question: CodeGenerationProblem, buggy_program: str, error_message: str):
    prompt = f"### Question:\n{question.question_content}\n\n"
    prompt += f"### Buggy program:\n```python\n{buggy_program}\n```\n\n"
    prompt += "You will also be provided with a control flow graph (CFG) image of this code. As you analyze each line:\n"
    prompt += "1. Examine each line of code sequentially to understand the overall structure and purpose of the code using both the code and CFG.\n"
    prompt += "2. Use this understanding to support your reasoning about the code's logic and potential errors.\n\n"
    prompt += f"### Error message:\n{error_message}\n\n"
    prompt += "Think through your analysis step by step, considering both the code and its representation in the CFG.\n\n"
    prompt += "After your analysis, provide a corrected version of the code that passes all tests.\n"
    prompt += "Your reasoning and corrected code: ....```python\n# YOUR CODE HERE\n```\n\n"
    return prompt

def get_VisualCoder_prompt(question: CodeGenerationProblem, buggy_program: str, error_message: str):
    prompt = f"### Question:\n{question.question_content}\n\n"
    prompt += f"### Buggy program:\n```python\n{buggy_program}\n```\n\n"
    prompt += f"### Error message:\n{error_message}\n\n"
    prompt += "You will also be provided with a control flow graph (CFG) image of this code. As you analyze each line:\n"
    prompt += "1. Examine each line of code sequentially and refer to the corresponding nodes in CFG to understand the code's logic and identify potential errors.\n"
    prompt += "2. Use this alignment between code and CFG to support your reasoning about the code's logic error and the way to fix it.\n\n"
    prompt += "Think through your analysis step by step, considering both the code and its representation in the CFG. After that, provide a corrected version of the code that passes all tests\n\n"
    prompt += "Your reasoning: ....```python\n# YOUR CODE HERE\n```\n\n"
    return prompt

def get_rationale_generation_prompt(problem, buggy_program, error_message):
    prompt = f"""You are an expert Python programmer.

You will be provided with a question (problem specification), a buggy Python program, and an error message. Please analyze the buggy program and generate a detailed rationale explaining why the program fails.

### Question:
{problem.question_content}

### Buggy program:
```python
{buggy_program}
```

### Error message:
{error_message}

You will also be provided with a control flow graph (CFG) image of this code. As you analyze each line:
1. Examine each line of code sequentially to understand the overall structure and purpose of the code using both the code and CFG.
2. Use this understanding to support your reasoning about the code's logic error and the way to fix it.

Please provide a detailed rationale explaining what might be wrong with the code and the way to fix it (no need the fixed code).""" 
    return prompt

def get_rationale_generation_VisualCoder_prompt(problem, buggy_program, error_message):
    prompt = f"""You will be provided with a question (problem specification), a buggy Python program, and an error message. Please analyze the buggy program and generate a detailed rationale explaining why the program fails.

### Question:
{problem.question_content}

### Buggy program:
```python
{buggy_program}
```

### Error message:
{error_message}

You will also be provided with a control flow graph (CFG) image of this code. As you analyze each line:
1. Examine each line of code sequentially and refer to the corresponding nodes in CFG to understand the code's logic and identify potential errors.
2. Use this alignment between code and CFG to support your reasoning about the code's logic error and the way to fix it.

Please provide a detailed rationale explaining what might be wrong with the code and the way to fix it(no need the corrected code).""" 
    return prompt

def get_answer_inference_prompt(problem, buggy_program, error_message, rationale):
    prompt = f"""You are an expert Python programmer.

You have been provided with a question (problem specification), a buggy Python program, its error message, and the following rationale for why the program fails:

### Question:
{problem}

### Rationale:
{rationale}

### Buggy program:
```python
{buggy_program}
```

### Error message:
{error_message}

Please use these information to fix the program so that it satisfies the problem's requirements. Only provide your corrected code within a Python code block (no need any explanation).

```python
# YOUR CORRECTED CODE HERE
```"""
    return prompt


def get_buggy_COT_prompt(question: CodeGenerationProblem, buggy_program: str, error_message: str):
    prompt = f"### Question:\n{question.question_content}\n\n"
    prompt += f"### Buggy program:\n```python\n{buggy_program}\n```\n\n"
    prompt += "Please follow these steps in your analysis:\n"
    prompt += "1. Examine each line of code sequentially to understand code.\n"
    prompt += "2. Use this understanding to support your reasoning about the code's logic and potential errors.\n\n"
    prompt += f"### Error message:\n{error_message}\n\n"
    prompt += "Think through your analysis step by step, considering the code's logic and potential errors.\n\n"
    prompt += "After your analysis, provide a corrected version of the code that passes all tests.\n"
    prompt += "Your reasoning and corrected code: ....```python\n# YOUR CODE HERE\n```\n\n"
    return prompt

def get_buggy_prompt(question: CodeGenerationProblem, buggy_program: str, error_message: str):
    prompt = f"### Question:\n{question.question_content}\n\n"
    prompt += f"### Buggy program:\n```python\n{buggy_program}\n```\n\n"
    prompt += f"### Error message:\n{error_message}\n\n"
    prompt += "Please identify the issues in the code and provide a corrected version that passes all tests.\n"
    prompt += "Your corrected code: ....```python\n# YOUR CODE HERE\n```\n\n"
    return prompt

def get_buggy_cfg_prompt(question: CodeGenerationProblem, buggy_program: str, error_message: str):
    prompt = f"### Question:\n{question.question_content}\n\n"
    prompt += f"### Buggy program:\n```python\n{buggy_program}\n```\n\n"
    prompt += f"### Error message:\n{error_message}\n\n"
    prompt += "You will also be provided with a control flow graph (CFG) image of this code."
    prompt += "Please analyze the CFG, identify the issues, and provide a corrected version of the code that passes all tests.\n"
    prompt += "Your corrected code: ....```python\n# YOUR CODE HERE\n```\n\n"
    return prompt

def get_next(question: CodeGenerationProblem, buggy_prorgam: str, error_message: str, trace: str):
    prompt = f"### Question:\n{question.question_content}\n\n"
    prompt += f"### Buggy program:\n```python\n{buggy_prorgam}\n```\n\n"
    prompt += f"### Error message:\n{error_message}\n\n"
    prompt += f"### Traces:\n```python\n{trace}\n```\n\n"
    prompt += "Please fix the code based on the trace and error message."
    prompt += "Your reasoning: ....```python\n# YOUR CODE HERE\n```\n\n"
    return prompt

def code_to_image(path: str):
    with open(path, 'rb') as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode('utf-8')


def run_image(problem: CodeGenerationProblem, buggy_program: str, error_message: str, image_data, trace: str):
    global openai_api_key
    
    if args.setting == "buggy_COT":
        prompt = get_buggy_COT_prompt(problem, buggy_program, error_message)
    elif args.setting == "buggy_cfg":
        prompt = get_buggy_cfg_prompt(problem, buggy_program, error_message)
    elif args.setting == "buggy_cfg_COT":
        prompt = get_buggy_cfg_COT_prompt(problem, buggy_program, error_message)
    elif args.setting == "buggy":
        prompt = get_buggy_prompt(problem, buggy_program, error_message)
    elif args.setting == "VisualCoder":
        prompt = get_VisualCoder_prompt(problem, buggy_program, error_message)
    elif args.setting == "NeXT":
        prompt = get_next(problem, buggy_program, error_message, trace)
    
    if args.close_model == "claude":
        client = anthropic.Anthropic(api_key=args.claude_api_key)
        chat_messages = [
            {
                "role": "user",
                "content": [  
                    { "type": "text", "text": prompt },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    },
                ]
            },
        ]
        try:
            response = client.messages.create(
                model=args.version,
                system=SYSTEM_MESSAGE_GENERIC,
                messages=chat_messages,
                max_tokens=2048
            )
            solution = response.content[0].text
        except:
            solution = ""
    elif args.close_model == "gpt":
        headers = {
            "Content-Type": "application/json",
            "api-key": args.openai_api_key
        }
        payload = {
            "model": args.version,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_MESSAGE_GENERIC,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        } if image_data else {},
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
            f"https://{args.azure_endpoint}/openai/deployments/{args.deployment_name}/chat/completions?api-version=2024-02-01",
            headers=headers,
            json=payload
        )
        solution = response.json()["choices"][0]["message"]["content"]


    match = re.search(pattern, solution, re.DOTALL)
    if match:
        python_code = match.group(1).strip()
    else:
        python_code = ""
    python_code = re.sub(r"assert.*|print.*", "", python_code)
    return solution, python_code

def run_code_repair_two_stage(problem: CodeGenerationProblem, buggy_program: str, error_message: str, image_data):
    global openai_api_key

    # Step 1: Rationale Generation (Stage 1)
    if args.setting == "Multimodal-CoT":
        rationale_prompt = get_rationale_generation_prompt(problem, buggy_program, error_message)
    elif args.setting == "Multimodal-CoT_VisualCoder":
        rationale_prompt = get_rationale_generation_VisualCoder_prompt(problem, buggy_program, error_message)

    # Generate rationale
    if args.close_model == "claude":
        client = anthropic.Anthropic(api_key=args.claude_api_key)
        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": rationale_prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    } if image_data else {},
                ]
            },
        ]
        try:
            response = client.messages.create(
                model=args.version,
                system="You are an expert Python programmer.",
                messages=chat_messages,
                max_tokens=1024
            )
            rationale = response.content[0].text
        except:
            rationale = ""
    elif args.close_model == "gpt":
        headers = {
            "Content-Type": "application/json",
            "api-key": args.openai_api_key
        }
        payload = {
            "model": args.version,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert Python programmer.",
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_data}"
                            }
                        } if image_data else {},
                        {
                            "type": "text",
                            "text": rationale_prompt,
                        }
                    ]
                }
            ],
            "max_tokens": 1024
        }
        response = requests.post(
            f"https://{args.azure_endpoint}/openai/deployments/{args.deployment_name}/chat/completions?api-version=2024-02-01",
            headers=headers,
            json=payload
        )
        rationale = response.json()["choices"][0]["message"]["content"]

    rationale = rationale.replace("```python", "").replace("```", "")
    
    # Step 2: Code Repair (Stage 2) using the generated rationale
    repair_prompt = get_answer_inference_prompt(problem, buggy_program, error_message, rationale)

    # Generate solution
    if args.close_model == "claude":
        chat_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": repair_prompt},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    } if image_data else {},
                ]
            },
        ]
        try:
            response = client.messages.create(
                model=args.version,
                system="You are an expert Python programmer.",
                messages=chat_messages,
                max_tokens=1024
            )
            solution = response.content[0].text
        except:
            solution = ""
    elif args.close_model == "gpt":
        payload["messages"][1]["content"] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}"
                }
            } if image_data else {},
            {
                "type": "text",
                "text": repair_prompt,
            }
        ]
        response = requests.post(
            f"https://{args.azure_endpoint}/openai/deployments/{args.deployment_name}/chat/completions?api-version=2024-02-01",
            headers=headers,
            json=payload
        )
        solution = response.json()["choices"][0]["message"]["content"]

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

generations_all = []
samples = []
responses_all = []
solutions_all = []
path = f"output/code_repair/{args.close_model}"
if not os.path.exists(path):
    os.makedirs(path)
path = os.path.join(path, f"{args.setting}_{args.session}.txt")

for idx in tqdm(range(33, len(dataset))):
    problem = CodeGenerationProblem(dataset[idx])
    generations = []
    solutions = []
    sample = problem.get_evaluation_sample()
    write(f"Problem {idx-33}", path)
    for j, buggy_program in enumerate(dataset[idx]["testable_programs"]):
        buggy_program = re.sub(r"assert.*|print.*", "", buggy_program)
        error_message = dataset[idx]["error_messages"][j]
        buggy_program = '\n'.join([line for line in buggy_program.split('\n') if not line.strip().startswith('#')])
        buggy_program = '\n'.join([line for line in buggy_program.split('\n') if not line.strip().startswith('@')])
        buggy_program = re.sub(r'(?<!["\'=#])\s*#.*$', '', buggy_program, flags=re.MULTILINE)
        filepath, _ = code2cfgimage(buggy_program, filename=f"coderepair_{args.setting}_{args.session}")
        image_data = code_to_image(filepath)
        write(f"Generation {j}", path)
        if args.setting == "Multimodal-CoT" or args.setting == "Multimodal-CoT_VisualCoder":
            rationale, solution, python_code = run_code_repair_two_stage(problem, buggy_program, error_message, image_data)
            write(f"Rationale: {rationale}", path)
            write(f"Solution:", path)
        else:
            if args.setting == "NeXT":
                trace = execute_and_trace(dataset[idx]["testable_programs"][j])
            else:
                trace = None
            solution, python_code = run_image(problem, buggy_program, error_message, image_data, trace)
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
