from datasets import load_dataset
from eval import evaluate_score_cruxeval
from prompts import make_cot_input_prompt, make_cot_output_prompt_cruxeval
from typing import Dict, Optional

class CruxEnv:
    def __init__(self, config):
        super().__init__()
        self.config = config
        limit = config.limit
        crux_dataset = load_dataset("cruxeval-org/cruxeval")
        crux_dataset = crux_dataset["test"]
        crux_dataset = crux_dataset.select(range(limit))
        self.crux_dataset = crux_dataset
        
        self.problem = None
        self.results = []
        
        self.make_prompt_func = make_cot_output_prompt_cruxeval if config.subset == "output" else make_cot_input_prompt
    
    @property
    def problems(self):
        return self.crux_dataset
    
    def set_problem(self, idx):
        self.problem = self.crux_dataset[idx]
        
    def get_problem_statement(self, make_prompt_func: Optional[callable]=None):
        example = self.problem
        s = (example["code"], example["input" if self.config.subset == "output" else "output"])
        if make_prompt_func is None:
            prompt = self.make_prompt_func(s)
        else:
            prompt = make_prompt_func(s)
        return prompt
    
    def get_code(self):
        example = self.problem
        return example["code"]
    
    def check_solution(self, solution):
        reference = (self.problem["code"], self.problem["input"], self.problem["output"])
        
        if self.config.subset == "input":
            if "[ANSWER]" in solution:
                solution = solution.split("[ANSWER]")[1].strip()
            if "==" in solution:
                solution = solution.split("==")[0].strip()
            if "assert f" in solution:
                solution = "f" + solution.split("assert f")[1].strip()
        else:
            if "[ANSWER]" in solution:
                solution = solution.split("[ANSWER]")[1].strip()
            if "\n[/ANSWER]" in solution:
                solution = solution.split("\n[/ANSWER]")[0].strip()
            if "==" in solution:
                solution = solution.split("==")[1].strip()  
        eval_arg = [solution], reference, self.config.subset
        results = evaluate_score_cruxeval(eval_arg)
        if results[0]:
            return True
        else:
            return False
        
    def extract_answers(self, proposed_solutions):
        answers = []
        for solution in proposed_solutions:
            if self.config.subset == "input":
                if "[ANSWER]" in solution:
                    solution = solution.split("[ANSWER]")[1].strip()
                if "==" in solution:
                    solution = solution.split("==")[0].strip()
                if "assert f" in solution:
                    solution = "f" + solution.split("assert f")[1].strip()
            else:
                if "[ANSWER]" in solution:
                    solution = solution.split("[ANSWER]")[1].strip()
                if "\n[/ANSWER]" in solution:
                    solution = solution.split("\n[/ANSWER]")[0].strip()
                if "==" in solution:
                    solution = solution.split("==")[1].strip()  
            answers.append(solution)
        return answers
    
    def accumulate_result(self, result: Dict):
        self.results.append(result)
        
    def finalize(self):
        self.result = {"correct": sum([result["is_correct"] for result in self.results]), "total": len(self.results), "error": sum([result["error"] for result in self.results])}