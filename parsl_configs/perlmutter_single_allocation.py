import parsl
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher, SingleNodeLauncher
from parsl.executors import HighThroughputExecutor


class PerlmutterConfig(Config):
    def __init__(self, run_config):
        gpu_executor = HighThroughputExecutor(
            label="gpu",
            max_workers_per_node=4,
            available_accelerators=4,
            provider=SlurmProvider(
                account=run_config["gpu_account"],
                qos=run_config["qos"],
                constraint="gpu",
                init_blocks=0,
                min_blocks=0,
                max_blocks=1,
                nodes_per_block=run_config["ngpu"],
                launcher=SrunLauncher(),
                scheduler_options="#SBATCH --gpus-per-node=4",
                walltime='24:00:00',
            )
        )

        cpu_executor = HighThroughputExecutor(
            label="cpu",
            max_workers_per_node=4,
            cores_per_worker=64,
            provider=SlurmProvider(
                account=run_config["cpu_account"],
                qos=run_config["qos"],
                constraint="cpu",
                init_blocks=0,
                min_blocks=1,
                max_blocks=1,
                nodes_per_block=run_config["ncpu"],
                launcher=SingleNodeLauncher(),
                walltime='24:00:00',
            )
        )

        super().__init__(
            executors=[gpu_executor, cpu_executor]
        )
