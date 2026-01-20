import parsl
from parsl_config_registry import CONFIG_REGISTRY


def load_parsl_config(run_config: dict):
    """
    Load and initialize a Parsl configuration from a registry.

    This function looks up a Parsl configuration class by name from
    ``CONFIG_REGISTRY`` and loads it using Parsl. If a Parsl DFK (DataFlowKernel)
    is already active, it clears the current Parsl context before loading
    the new configuration.

    Parameters
    ----------
    run_config : dict
        Run configuration dictionary. Must contain key ``"parsl_config"`` whose
        value is a string matching a key in ``CONFIG_REGISTRY``. Additional keys
        are passed to the configuration class constructor.

    Returns
    -------
    parsl.config.Config
        The loaded Parsl configuration object returned by ``parsl.load(...)``.

    Raises
    ------
    KeyError
        If ``run_config["parsl_config"]`` is not found in ``CONFIG_REGISTRY``.

    Notes
    -----
    - The configuration class is instantiated as ``cls(run_config)``.
    - If ``parsl.dfk()`` returns a non-None object, ``parsl.clear()`` is called
      before loading the new configuration.
    """
    config_name = run_config["parsl_config"]
    if config_name not in CONFIG_REGISTRY:
        raise KeyError(
            f"Unknown config '{config_name}'. Available: {', '.join(CONFIG_REGISTRY)}"
        )

    cls = CONFIG_REGISTRY[config_name]
    if parsl.dfk() is not None:
        parsl.clear()
    return parsl.load(cls(run_config))
