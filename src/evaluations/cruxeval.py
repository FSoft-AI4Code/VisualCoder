from src.utils.utils import check_correctness

def evaluate_score_cruxeval(args):
    gs, (c, i, o), mode = args

    execution_results = []
    for g in gs:
        if mode == "input" and "f(" not in g:
            pass
        elif mode == "output" and f"f({i})" in g:
            pass
        else:
            code_to_execute = f"{c}\nassert {o} == {g}"
            execution_results.append(check_correctness(code_to_execute, 3))
    if True not in execution_results:
        execution_results = [False] * len(gs)
    return execution_results