from src.tasks.execution.crux_execution import CruxEvalTask

def get_task(config):
    if config.task == "cruxeval":
        return CruxEvalTask(config)
    else:
        raise NotImplementedError(f"Task {config.task} not implemented.")
    