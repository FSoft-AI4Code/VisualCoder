from datasets import load_dataset
import json
from dataclasses import dataclass
from enum import Enum
import zlib
import pickle
from src.utils.utils import evaluate_generations, codegen_metrics
import os
import tqdm 
import anthropic
import re
import argparse
import random

class PromptConstants:
    SYSTEM_MESSAGE_GENERIC = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You should think step-by-step logically before returning final the program. The program should only include function definition with parameter list in order."

    SYSTEM_MESSAGE_GEMINI = f"You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. Do NOT use system calls like `exit` in the generated program."

    SYSTEM_MESSAGE_DEEPSEEK = f"You are an AI programming assistant, utilizing the DeepSeek Coder model, developed by DeepSeek Company, and you answer questions related to computer science."

    SYSTEM_MESSAGE_MAGIC = f"You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.\n\n@@ Instruction\n"

    SYSTEM_MESSAGE_WIZARD = "Below is an instruction that describes a task. Write a response that appropriately completes the request."

    SYSTEM_MESSAGE_PHIND = f"""You are an expert Python programmer. You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program. Put your fixed program within code delimiters, for example: 
```python 
# YOUR CODE HERE
```"""

    SYSTEM_MESSAGE_CODEQWEN = (
        f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user"
    )

    FORMATTING_MESSAGE_WITH_STARTER_CODE = "You will use the following starter code to write the solution to the problem and enclose your code within delimiters."

    FORMATTING_WITHOUT_STARTER_CODE = "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows."
    
class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"

@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)
        
class CodeGenerationProblem:
    def __init__(self, data_dict: dict):
        self.question_content = data_dict["question_content"]
        self.starter_code = data_dict["starter_code"] if len(data_dict["starter_code"]) != 0 else None
        self.public_test_cases = json.loads(data_dict["public_test_cases"])
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]
        self.metadata = json.loads(data_dict["metadata"]) if "metadata" in data_dict else {}
        
        try:
            self.private_test_cases = json.loads(data_dict["private_test_cases"])  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(data_dict["private_test_cases"].encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]
        
    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }
    
def get_generic_question_template_answer(question: CodeGenerationProblem):
    prompt = f"### Question:\n{question.question_content}\n\n"
    if question.starter_code:
        prompt += (
            f"### Format: {PromptConstants.FORMATTING_MESSAGE_WITH_STARTER_CODE}\n"
        )
        prompt += f"```python\n{question.starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {PromptConstants.FORMATTING_WITHOUT_STARTER_CODE}\n"
        prompt += "Your reasoning: ....```python\n# YOUR CODE HERE\n```\n\n"
    prompt += f"### Answer: (use the provided format with backticks)\n\n"
    return prompt


pattern = r"```python(.*?)```"  
def run(problem: CodeGenerationProblem, client: anthropic.Anthropic, num_samples: int = 5, mix=False):
    generations = []
    raw_responses = []
    chat_messages = [
        {
            "role": "user",
            "content": get_generic_question_template_answer(problem),
        },
    ]
    for i in range(num_samples):
        if not mix:
            response = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                system=PromptConstants.SYSTEM_MESSAGE_GENERIC,
                messages=chat_messages,
                max_tokens=2048
            )
            solution = response.content[0].text
        else:
            try:
                if random.random() < 0.75:
                    response = client.messages.create(
                        model="claude-3-5-sonnet-20240620",
                        system=PromptConstants.SYSTEM_MESSAGE_GENERIC,
                        messages=chat_messages,
                        max_tokens=2048
                    )
                    solution = response.content[0].text
                else:
                    response = client.messages.create(
                        model="claude-3-haiku-20240307",
                        system=PromptConstants.SYSTEM_MESSAGE_GEMINI,
                        messages=chat_messages,
                        max_tokens=2048
                    )
                    solution = response.content[0].text
            except:
                solution = ""
        match = re.search(pattern, solution, re.DOTALL)
        if match:
            python_code = match.group(1).strip()
            generations.append(python_code)
            raw_responses.append(solution)
    return generations, raw_responses


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--mix", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = arg_parser()
    
    dataset = load_dataset("livecodebench/code_generation")
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    
    generations_all = []
    samples = []
    responses_all = []

    for idx in tqdm.tqdm(range(len(dataset["test"]))):
    # for idx in tqdm.tqdm(range(1)):
        problem = CodeGenerationProblem(dataset["test"][idx])
        generations, raw_responses = run(problem, client, num_samples=args.num_samples, mix=args.mix)
        generations_all.append(generations)
        responses_all.append(raw_responses)
        samples.append(problem.get_evaluation_sample())
    
    with open("temp.pkl", "wb") as f:
        pickle.dump((generations_all, samples, responses_all), f)
        
    results, metadata = evaluate_generations(samples, generations_all, timeout=6)
    
    
    metric = codegen_metrics(samples, generations_all)
    metrics, results, final_metadata = metric[0], metric[1], metric[2]

    instances = {}

    for idx, result_all_samples in results.items():
        for sol_id, result in enumerate(result_all_samples):
            if False in result:
                if idx not in instances:
                    instances[idx] = []
                instances[idx].append({"failed_solution": generations_all[idx][sol_id], "failed_test_cases": [i for i, r in enumerate(result) if not r]})
                print(f"Problem {idx} failed with solution {sol_id} with test cases {instances[idx][-1]['failed_test_cases']}")

    buggy_dataset = dataset["test"].add_column(name="failed_generations", column=[instances[i] if i in instances else None for i in range(len(dataset["test"]))])
    buggy_dataset = buggy_dataset.filter(lambda x: x["failed_generations"] is not None)
    buggy_dataset.save_to_disk("buggy_dataset.json")
    buggy_dataset.push_to_hub("huypn16/LCB-R")