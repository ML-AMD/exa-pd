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
