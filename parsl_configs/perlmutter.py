import parsl
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher, SingleNodeLauncher
from parsl.executors import HighThroughputExecutor


class PerlmutterConfig(Config):
    """
    Parsl configuration for the NERSC Perlmutter system.

    This configuration defines two Parsl executors backed by Slurm:
    - a GPU executor for running LAMMPS GPU jobs, and
    - a CPU executor for running LAMMPS CPU jobs.

    Resource allocation (number of nodes, scheduler options, etc.) is
    controlled by the ``run_config`` dictionary provided at construction
    time, typically read from the input JSON file.

    Parameters
    ----------
    run_config : dict
        Runtime configuration dictionary. Expected keys include:

        - ``ngpu`` : int
            Number of GPU nodes per Slurm block.
        - ``ncpu`` : int
            Number of CPU nodes per Slurm block.
        - ``gpu_schedule_option`` : list of str
            Slurm scheduler directives for GPU jobs.
        - ``cpu_schedule_option`` : list of str
            Slurm scheduler directives for CPU jobs.

    Notes
    -----
    - Both executors use :class:`HighThroughputExecutor`.
    - GPU jobs are launched using :class:`SrunLauncher`.
    - CPU jobs are launched using :class:`SingleNodeLauncher`.
    - ``init_blocks`` and ``min_blocks`` are set to zero so resources
      are provisioned on demand.
    """

    def __init__(self, run_config):
        """
        Initialize the Perlmutter Parsl configuration.

        Parameters
        ----------
        run_config : dict
            Runtime configuration dictionary.
        """
        gpu_executor = HighThroughputExecutor(
            label="gpu",
            max_workers_per_node=4,
            available_accelerators=4,
            provider=SlurmProvider(
                init_blocks=0,
                min_blocks=0,
                max_blocks=10,
                nodes_per_block=run_config["ngpu"],
                launcher=SrunLauncher(),
                scheduler_options="\n".join(run_config["gpu_schedule_option"]),
            )
        )

        cpu_executor = HighThroughputExecutor(
            label="cpu",
            max_workers_per_node=4,
            cores_per_worker=64,
            provider=SlurmProvider(
                init_blocks=0,
                min_blocks=0,
                max_blocks=10,
                nodes_per_block=run_config["ncpu"],
                launcher=SingleNodeLauncher(),
                scheduler_options="\n".join(run_config["cpu_schedule_option"]),
            )
        )

        super().__init__(
            executors=[gpu_executor, cpu_executor]
        )
