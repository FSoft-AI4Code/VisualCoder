import sys
import trace_execution
import inspect
from io import StringIO
from contextlib import redirect_stdout

class ExecutionTracer:
    def __init__(self):
        self.tracer = trace_execution.Trace(count=False, trace=True, countfuncs=False, countcallers=False)
        self.execution_trace = []

    def trace_execution(self, frame, event, arg):
        if event == 'line':
            code = frame.f_code
            lineno = frame.f_lineno
            locals_snapshot = frame.f_locals.copy()
            self.execution_trace.append((lineno, locals_snapshot))
        return self.trace_execution

    def start_tracing(self, func):
        sys.settrace(self.trace_execution)
        func()
        sys.settrace(None)

    def get_execution_trace(self):
        return self.execution_trace

def filter_execution_trace(execution_trace):
    filtered_trace = []
    for lineno, vars_at_line in execution_trace:
        # Skip entries with 'code' or function names in variables
        if 'code' in vars_at_line or any(callable(var) for var in vars_at_line.values()):
            continue
        if '__builtins__' in vars_at_line:
            vars_at_line.pop('__builtins__')  # Remove built-ins as they're not needed
        if len(vars_at_line) > 0:  # Only keep entries with meaningful local variables
            filtered_trace.append((lineno, vars_at_line))
    return filtered_trace

def add_inline_trace(file_path, execution_trace):
    with open(file_path, 'r') as file:
        code_lines = file.readlines()

    traced_code = []
    last_lineno = -1
    for lineno, local_vars in execution_trace:
        while last_lineno < lineno - 1:
            last_lineno += 1
            traced_code.append(code_lines[last_lineno].rstrip())

        comment = f" # ({lineno}) " + "; ".join([f"{var}={val}" for var, val in local_vars.items()])
        traced_code[-1] += comment

    while last_lineno < len(code_lines) - 1:
        last_lineno += 1
        traced_code.append(code_lines[last_lineno].rstrip())

    return "\n".join(traced_code)

def execute_and_trace(file_path):
    with open(file_path, 'r') as file:
        code = compile(file.read(), file_path, 'exec')
    
    def run_code():
        exec(code, {})
    
    tracer = ExecutionTracer()
    tracer.start_tracing(run_code)

    # Clean up the trace to remove irrelevant initial entries
    cleaned_trace = filter_execution_trace(tracer.get_execution_trace())

    traced_code = add_inline_trace(file_path, cleaned_trace)

    # Write the traced code with comments to a new file
    with open(f'traced_{file_path}', 'w') as traced_file:
        traced_file.write(traced_code)

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python tracer.py <python_file_path>")
    else:
        execute_and_trace(sys.argv[1])
        print(f"Traced file saved as traced_{sys.argv[1]}")
