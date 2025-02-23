import sys
import src.codetransform.trace_execution as trace_execution
import copy
import linecache
import random

class ExecutionTracer:
    def __init__(self):
        self.tracer = trace_execution.Trace(count=False, trace=True, countfuncs=False, countcallers=False)
        self.execution_trace = []
        self.idx = 0

    def trace_execution(self, frame, event, arg):
        if '__builtins__' not in frame.f_locals and ".0" not in frame.f_locals and len(frame.f_locals) > 0 and 'code' not in frame.f_locals:
            code = frame.f_code
            lineno = frame.f_lineno
            locals_snapshot = copy.deepcopy(frame.f_locals) 
            self.execution_trace.append([lineno, locals_snapshot])
        return self.trace_execution

    def start_tracing(self, func):
        sys.settrace(self.trace_execution)
        func()
        sys.settrace(None)

    def get_execution_trace(self):
        return self.execution_trace

def generate_commented_code(file_path, in_line_cmt):

    with open(file_path, 'r') as file:
        code_lines = file.readlines()
        code_lines = [element.rstrip() for element in code_lines]
    commented_code = []

    for lineno, line in enumerate(code_lines, 1):
        if lineno in in_line_cmt:
            comments = in_line_cmt[lineno]
            
            if comments:
                first_comment = comments[0]
                last_comment = comments[-1]

                # Handle NO_CHANGE scenario
                if isinstance(first_comment[1], str) and first_comment[1] == "NO_CHANGE":
                    first_comment_str = f"({first_comment[0]}) NO_CHANGE"
                else:
                    first_comment_str = f"({first_comment[0]}) " + "; ".join([f"{k}={v}" for k, v in first_comment[1].items()])
                
                if first_comment == last_comment:
                    inline_comment = f" # {first_comment_str}"
                else:
                    if isinstance(last_comment[1], str) and last_comment[1] == "NO_CHANGE":
                        last_comment_str = f"({last_comment[0]}) NO_CHANGE"
                    else:
                        last_comment_str = f"({last_comment[0]}) " + "; ".join([f"{k}={v}" for k, v in last_comment[1].items()])
                    
                    inline_comment = f" # {first_comment_str}; ...; {last_comment_str}"
                
                commented_code.append(line + inline_comment)
            else:
                commented_code.append(line)
        else:
            commented_code.append(line)

    # Special handling for return statements if necessary
    for i, line in enumerate(commented_code):
        if 'return' in line:
            return_line = in_line_cmt.get(lineno, [])
            if return_line:
                return_vars = return_line[0][1]
                if isinstance(return_vars, str) and return_vars == "NO_CHANGE":
                    return_comment = "NO_CHANGE"
                else:
                    return_comment = "; ".join([f"{k}={v}" for k, v in return_vars.items()])
                commented_code[i] = f"{line} # __return__=({return_comment})"

    return "\n".join(commented_code)

def generate_random_string(input_string, length=64):
    # Ensure input_string is not empty to avoid ValueError
    if not input_string:
        raise ValueError("Input string must not be empty.")
    
    # Generate a random string of the specified length
    random_string = ''.join(random.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(length))
    return random_string

def execute_and_trace(source: str):
    # generate a random temp file name
    file_path = generate_random_string(source, 32)
    file_path += ".py"
    file_path = f"/tmp/{file_path}"
    
    with open(file_path, 'w') as file:
        file.write(source)
    
    with open(file_path, 'r') as file:
        code = compile(file.read(), file_path, 'exec')

    def run_code():
        try:
            exec(code, {})
        except:
            pass

    tracer = ExecutionTracer()
    tracer.start_tracing(run_code)
    execution_trace = tracer.get_execution_trace()

    intermediate_value = []
    source = linecache.getlines(file_path)
    code_line = [element.lstrip().replace('\n', '') for element in source]
    condition_line = []

    for i in range(len(code_line)):
        if code_line[i].find('if') != -1 or code_line[i].find('elif') != -1 or code_line[i].find('else') != -1:
            condition_line.append(i+1)
    for i in range(len(execution_trace)-1):
        if execution_trace[i][0] not in condition_line:
            intermediate_value.append([execution_trace[i][0], execution_trace[i+1][1]])
        
    symbol_table = {}
    values = []
    for i in range(len(intermediate_value)):
        if i == len(intermediate_value)-1:
            values.append([intermediate_value[i][0], intermediate_value[i][1]])
        else:
            temp_dict = {}
            for var in intermediate_value[i][1]:
                if var not in symbol_table:
                    symbol_table[var] = intermediate_value[i][1][var]
                    temp_dict[var] = intermediate_value[i][1][var]
                else:
                    if symbol_table[var] != intermediate_value[i][1][var]:
                        symbol_table[var] = intermediate_value[i][1][var]
                        temp_dict[var] = intermediate_value[i][1][var]
            if len(temp_dict) > 0:
                values.append([intermediate_value[i][0], temp_dict])
            else:
                values.append([intermediate_value[i][0], "NO_CHANGE"])

    in_line_cmt = {}
    for i in range(len(values)):
        if values[i][0] not in in_line_cmt:
            in_line_cmt[values[i][0]] = [[i, values[i][1]]]
        else:
            in_line_cmt[values[i][0]].append([i, values[i][1]])
        
    commented_code = generate_commented_code(file_path, in_line_cmt)

    return commented_code


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python tracer.py <python_file_path>")
    else:
        execute_and_trace(sys.argv[1])
        print(f"Traced file saved as traced_{sys.argv[1]}")
