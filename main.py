import time 
from pydantic import BaseModel
from tqdm import tqdm
from src.tasks.build import get_task
from src.generations.build import get_run_engine
from src.codetransform.cfg2image import code2cfg

class config(BaseModel):
    task: str = "cruxeval"
    limit: int = 250
    subset: str = "output"


if __name__ == "__main__":
    cfg = config()
    task = get_task(cfg)
    run_engine = get_run_engine(cfg)
    
    for idx in tqdm(range(1)):
        error = False
        task.set_problem(idx)
        prompt = task.get_problem_statement()
        code = task.get_code()
        
        # Initialize variables
        attempts = 0
        max_attempts = 5
        while attempts < max_attempts:
            try:
                solution = get_run_engine(prompt, code2cfg(code))
                break 
            except Exception as e:
                print(e)
                error = True
                attempts += 1  # Increment the attempts counter
                if attempts < max_attempts:
                    time.sleep(60)  
                else:
                    solution = ""  
                    print("Failed after 4 attempts")
                    break 
        correct = task.check_solution(solution)
        print(correct)
        task.accumulate_result({"is_correct": correct, "solution": solution, "error": error})
    task.finalize()
    print(task.result)