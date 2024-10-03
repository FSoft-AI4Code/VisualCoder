import re

def parse_steps(response_text):
    """
    Parses the response text to identify steps using regex.
    Assumes steps are formatted like '1. **Step Title:**'
    """
    # Corrected regex pattern to capture steps like '1. **Initialization:**'
    step_pattern = re.compile(r'(\d+)\.\s+\*\*(.*?):\*\*', re.MULTILINE)
    steps = []
    
    # Find all matches
    matches = list(step_pattern.finditer(response_text))
    
    for i, match in enumerate(matches):
        step_num = int(match.group(1))
        step_title = match.group(2).strip()
        start_char = match.start()  # Start from the beginning of the title
        if i < len(matches) - 1:
            end_char = matches[i + 1].start()
        else:
            end_char = len(response_text)
        
        # Extract the content for this step, including the title
        content = response_text[start_char:end_char].strip()
        
        steps.append({
            'num': step_num,
            'title': step_title,
            'start_char': start_char,
            'end_char': end_char,
            'content': content
        })
    
    return steps


text = """Let's analyze the code step by step and compare it with the control flow graph (CFG) to identify the problematic line.

1. **Initialization:**
   ```python
   S = ['1', '2', '1', '4']
   K = 4
   ```
   - This line initializes the list `S` to `['1', '2', '1', '4']` and the variable `K` to 4. This corresponds to the node at the top of the CFG.

2. **For loop:**
   ```python
   for i in range(len(K)):
       if S[i]!= 1:
           print(S[i])
           exit()
       print(1)
   ```
   - This line iterates over the indices of `S` using the length of `K` (which is 4). This corresponds to the `for` loop node in the CFG.

3. **First `if` statement:**
   ```python
   if S[i]!= 1:
       print(S[i])
       exit()
   ```
   - This line checks if the element at index `i` in `S` is not equal to 1. This corresponds to the first `if` node in the CFG.

4. **Second `if` statement:**
   ```python
   print(1)
   ```
   - This line prints 1. This corresponds to the second `if` node in the CFG.

5. **Third `if` statement:**
   ```python
   if S[i]!= 1:
       print(S[i])
       exit()
   ```
   - This line checks if the element at index `i` in `S` is not equal to 1. This corresponds to the third `if` node in the CFG.

6. **Fourth `if` statement:**
   ```python
   print(1)
   ```
   - This line prints 1. This corresponds to the fourth `if` node in the CFG.

Now, let's identify the problematic line:

- The first `if` statement checks if the element at index `i` in `S` is not equal to 1. This corresponds to the first `if` node in the CFG.

- The second `if` statement prints 1. This corresponds to the second `if` node in the CFG.

- The third `if` statement checks if the element at index `i` in `S` is not equal to 1. This corresponds to the third `if` node in the CFG.

- The fourth `if` statement prints 1. This corresponds to the fourth `if` node in the CFG."""

if __name__ == '__main__':
    steps = parse_steps(text)
    for step in steps:
        print(f"Step {step['num']}: {step['content']}")