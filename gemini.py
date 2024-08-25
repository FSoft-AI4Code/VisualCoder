import time 
from pydantic import BaseModel
from tqdm import tqdm
from env import CruxEnv
from py2cfg import CFGBuilder
# Gemini 
import google.generativeai as genai 
import PIL.Image 
import os

def visual_run_gemini(client, prompt, image_path):
    img = PIL.Image.open(image_path)
    prompt = "You are given a control flow graph image of a code snippet, utilize them in code execution reasoning process. " + prompt
    return client.generate_content([img, prompt]).text

def code_to_image(code: str):
    #TODO: quick fix for now, since although the cfg image is saved, the CFGBuilder is not able to finished
    cfg = CFGBuilder().build_from_src('ControlFlowGraph', code)
    render = cfg.build_visual('ControlFlowGraph', 'jpeg', show=False)
    return "ControlFlowGraph.jpeg"

class config(BaseModel):
    limit: int = 250
    subset: str = "output"

def make_visual_cot_output_prompt(s):
    code, input = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(L, m, start, step): 
    L.insert(start, m) 
    for x in range(start-1, 0, -step): 
        start -= 1
        L.insert(start, L.pop(L.index(m)-1)) 
    return L
assert f(thigh_o_two[:], 3, 3, 2) == ??
[/PYTHON]
[THOUGHT]
Let's execute code step by step:
1. Initial State:
	L = [1, 2, 7, 9]
	m = 3
	start = 3
	step = 2
2. First Operation (L.insert(start, m)):
	This is shown in the control flow graph as the first action after the function begins.
	Insert m (which is 3) at index start (which is 3).
	L = [1, 2, 7, 3, 9]
3. For Loop Initialization (for x in range(start - 1, 0, -step)):
	range(start - 1, 0, -step) becomes range(2, 0, -2) because start is 3.
	The loop will iterate with x taking values 2.
	The control flow graph indicates this loop.
4. First Loop Iteration (x = 2):
	Decrement start by 1: start = start - 1 = 2.
	L.pop(L.index(m) - 1):
	L.index(m) finds the index of m (which is 3) in L. The index of 3 is 3.
	L.index(m) - 1 is 3 - 1 = 2.
	L.pop(2) removes and returns the element at index 2, which is 7.
	L.insert(start, 7):
	Insert 7 at index start (which is 2).
	L becomes [1, 2, 7, 3, 9] after removing 7 and inserting it back at the same position. (No visible change)
5. End of Loop:
	The range range(2, 0, -2) has no more values after x = 2, so the loop ends.

After following the control flow of the function and given input parameters, the final output is: [1, 2, 7, 3, 9]
[/THOUGHT]
[ANSWER]
f(thigh_o_two[:], 3, 3, 2) == [1, 2, 7, 3, 9]
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[THOUGHT]
"""

if __name__ == "__main__":
    genai.configure(api_key="API_HERE") 
    client = genai.GenerativeModel(model_name="gemini-1.5-flash") 
    cfg = config()
    env = CruxEnv(cfg)
    
    for idx in tqdm(range(1)):
        error = False
        env.set_problem(idx)
        prompt = env.get_problem_statement(make_visual_cot_output_prompt)
        code = env.get_code()
        
        # Initialize variables
        attempts = 0
        max_attempts = 5
        while attempts < max_attempts:
            try:
                solution = visual_run_gemini(client, prompt, code_to_image(code))
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
        correct = env.check_solution(solution)
        print(correct)
        env.accumulate_result({"is_correct": correct, "solution": solution, "error": error})
    env.finalize()
    print(env.result)