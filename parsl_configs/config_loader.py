import parsl
from parsl_config_registry import CONFIG_REGISTRY

def load_parsl_config(run_config: dict):
    config_name = run_config["parsl_config"]
    if config_name not in CONFIG_REGISTRY:
        raise KeyError(f"Unknown config '{config_name}'. Available: {', '.join(CONFIG_REGISTRY)}")

    cls = CONFIG_REGISTRY[config_name]
    if parsl.dfk() is not None:
        parsl.clear()
    return parsl.load(cls(run_config))