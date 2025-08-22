import parsl
import importlib
from tools.logging_config import exapd_logger

CONFIG_REGISTRY = {
    "perlmutter": "parsl_configs.perlmutter.PerlmutterConfig",
    "perlmutter_single_alloc": "parsl_configs.perlmutter_single_allocation.PerlmutterConfig"
}

def load_parsl_config(run_config: dict):
    config_name = run_config["parsl_config"]
    if config_name not in CONFIG_REGISTRY:
        exapd_logger.critical(f"Unknown config '{config_name}'. Available: {', '.join(CONFIG_REGISTRY)}")

    selector = CONFIG_REGISTRY[config_name]
    module_path, class_name = selector.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return parsl.load(cls(run_config))