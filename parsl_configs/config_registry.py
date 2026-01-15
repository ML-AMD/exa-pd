import parsl
import importlib
from tools.logging_config import exapd_logger

# Registry mapping configuration names to fully-qualified class paths
CONFIG_REGISTRY = {
    "perlmutter": "parsl_configs.perlmutter.PerlmutterConfig",
    "perlmutter_single_alloc": "parsl_configs.perlmutter_single_allocation.PerlmutterConfig"
}


def load_parsl_config(run_config: dict):
    """
    Load and initialize a Parsl configuration from a string registry.

    This function selects a Parsl configuration class based on the
    ``parsl_config`` entry in ``run_config``, dynamically imports the
    corresponding module, and loads the configuration into Parsl.

    Parameters
    ----------
    run_config : dict
        Runtime configuration dictionary. Must contain key
        ``"parsl_config"`` specifying one of the keys in ``CONFIG_REGISTRY``.

    Returns
    -------
    parsl.config.Config
        The loaded Parsl configuration instance returned by ``parsl.load``.

    Raises
    ------
    SystemExit
        If an unknown configuration name is provided. Program termination
        occurs via :func:`exapd_logger.critical`.

    Notes
    -----
    - Configuration classes are expected to accept ``run_config`` as their
      sole constructor argument.
    - Dynamic imports are performed using :mod:`importlib`.
    """
    config_name = run_config["parsl_config"]
    if config_name not in CONFIG_REGISTRY:
        exapd_logger.critical(
            f"Unknown config '{config_name}'. Available: {', '.join(CONFIG_REGISTRY)}"
        )

    selector = CONFIG_REGISTRY[config_name]
    module_path, class_name = selector.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    return parsl.load(cls(run_config))
