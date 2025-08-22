import parsl
from parsl import python_app, bash_app
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher, SimpleLauncher, SingleNodeLauncher
from parsl.executors import HighThroughputExecutor
from logging_config import exapd_logger

class PerlmutterConfig(Config):
    def __init__(self,run_config):
        gpu_executor = HighThroughputExecutor(
            label="gpu",
            max_workers_per_node=4,
            available_accelerators=4,
            provider=SlurmProvider(
                account=run_config["gpu_account"],
                qos=run_config["qos"],
                constraint="gpu",
                init_blocks=0,
                min_blocks=1,
                max_blocks=10,
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
                max_blocks=10,
                nodes_per_block=run_config["ncpu"],
                launcher=SingleNodeLauncher(),
                walltime='24:00:00',
                )
            )

        super().__init__(
            executors=[gpu_executor, cpu_executor]
            )


@bash_app(executors=["gpu"])
def gpu_lammps(directory, script, lmp_exe, run_para="", stdout=parsl.AUTO_LOGNAME, stderr=parsl.AUTO_LOGNAME):
    return f"""
    dir="{directory}"
    cd {directory}
    echo "Running command: {lmp_exe} -sf gpu -pk gpu 1 -in {script} {run_para} > lmp.out"
    {lmp_exe} -sf gpu -pk gpu 1 -in {script} {run_para} > lmp.out
    touch DONE
    exapd_logger.info(f"Done job: {directory}")
    """

@bash_app(executors=["cpu"])
def cpu_lammps(directory, script, lmp_exe,run_para="", ncpu=32, stdout=parsl.AUTO_LOGNAME, stderr=parsl.AUTO_LOGNAME):
    return f"""
    dir="{directory}"
    cd {directory}
    echo "Running command: srun -N 1 -n {ncpu} -c 2 --cpu-bind=cores --exact shifter --image=nersc/lammps_lite:23.08 {lmp_exe} -in {script} {run_para} > lmp.out"
    srun -N 1 -n {ncpu} -c 2 --cpu-bind=cores --exact shifter --image=nersc/lammps_lite:23.08 {lmp_exe} -in {script} {run_para} > lmp.out
    touch DONE
    exapd_logger.info(f"Done job: {directory}")
    """

'''

@python_app(executors=["gpu"])
def gpu_lammps(directory, script, lmp_exe):
    import subprocess
    import os
    print(f"Running lmp: {directory}")
    subprocess.run(f"cd {directory} && {lmp_exe} -sf gpu -pk gpu 1 neigh no -in {script} > lmp.out", shell=True)
    return

@python_app(executors=["cpu"])
def cpu_lammps(directory, script, ncpu=32):
    import subprocess
    import os
    print(f"Running lmp: {directory}")
    subprocess.run(f"cd {directory} && srun -N 1 -n {ncpu} -c 2 --cpu-bind=cores --exact shifter --image=nersc/lammps_lite:23.08 lmp -in {script} > lmp.out", shell=True)
    return

'''
