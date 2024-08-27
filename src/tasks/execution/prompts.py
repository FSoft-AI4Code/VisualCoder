def make_cot_output_prompt_cruxeval(s):
    code, input = s
    return f"""You are given a Python function and an assertion containing an input to the function. Complete the assertion with a literal (no unsimplified expressions, no function calls) containing the output when executing the provided code on the given input, even if the function is incorrect or incomplete. Do NOT output any extra information. Execute the program step by step before arriving at an answer, and provide the full assertion with the correct output in [ANSWER] and [/ANSWER] tags, following the examples.

[PYTHON]
def f(s):
    s = s + s
    return "b" + s + "a"
assert f("hi") == ??
[/PYTHON]
[THOUGHT]
Let's execute the code step by step:

1. The function f is defined, which takes a single argument s.
2. The function is called with the argument "hi", so within the function, s is initially "hi".
3. Inside the function, s is concatenated with itself, so s becomes "hihi".
4. The function then returns a new string that starts with "b", followed by the value of s (which is now "hihi"), and ends with "a".
5. The return value of the function is therefore "bhihia".
[/THOUGHT]
[ANSWER]
assert f("hi") == "bhihia"
[/ANSWER]

[PYTHON]
{code}
assert f({input}) == ??
[/PYTHON]
[THOUGHT]
"""

def make_cot_input_prompt(s):
    code, output = s
    return f"""You will be given a function f and an output in the form f(??) == output. Your task is to find any input such that executing f on the input leads to the given output. There may be multiple answers, but only output one. First, think step by step. You MUST surround the answer with [ANSWER] and [/ANSWER] tags. Express your answer as a passing assertion containing the input and the given output.

[PYTHON]
def f(x):
    return x + 1
assert f(??) == 17
[/PYTHON]
[THOUGHT]
To find an input such that executing f on the input leads to the given output, we can work backwards from the given assertion. We know that f(??) == 17. 

Since the function f(x) returns x + 1, for f(??) to be equal to 17, the value of ?? should be 16. 
[/THOUGHT]
[ANSWER]
assert f(16) == 17
[/ANSWER]

[PYTHON]
{code}
assert f(??) == {output}
[/PYTHON]
[THOUGHT]
"""

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