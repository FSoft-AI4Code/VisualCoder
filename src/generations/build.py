from src.generations.execution import run_model_execution

def get_run_engine(config):
    if config.engine == "execution":
        return run_model_execution
    else:
        raise NotImplementedError(f"Engine {config.engine} not implemented")