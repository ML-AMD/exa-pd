import parsl
from parsl import python_app, bash_app
from parsl.config import Config
from parsl.providers import SlurmProvider
from parsl.launchers import SrunLauncher, SimpleLauncher, SingleNodeLauncher
from parsl.executors import HighThroughputExecutor

class PerlmutterConfig(Config):
    def __init__(self):
        gpu_executor = HighThroughputExecutor(
            label="gpu",
            max_workers_per_node=4,
            available_accelerators=4,
            provider=SlurmProvider(
                account="m4802",
                qos="premium",
                constraint="gpu",
                init_blocks=1,
                max_blocks=10,
                nodes_per_block=1,
                launcher=SrunLauncher(),
                scheduler_options="#SBATCH --gpus=4",
                walltime='24:00:00',
                )
            )

        cpu_executor = HighThroughputExecutor(
            label="cpu",
            max_workers_per_node=4,
            cores_per_worker=64,
            provider=SlurmProvider(
                account="m4802",
                qos="premium",
                constraint="cpu",
                init_blocks=1,
                max_blocks=10,
                nodes_per_block=1,
                #launcher=SimpleLauncher(),
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
    {lmp_exe} -sf gpu -pk gpu 1 -in {script} {run_para} > lmp.out
    """

@bash_app(executors=["cpu"])
def cpu_lammps(directory, script, run_para="", ncpu=32, stdout=parsl.AUTO_LOGNAME, stderr=parsl.AUTO_LOGNAME):
    return f"""
    dir="{directory}"
    cd {directory}
    echo "Running command: srun -N 1 -n {ncpu} -c 2 --cpu-bind=cores --exact shifter --image=nersc/lammps_lite:23.08 lmp -in {script} {run_para} > lmp.out"
    srun -N 1 -n {ncpu} -c 2 --cpu-bind=cores --exact shifter --image=nersc/lammps_lite:23.08 lmp -in {script} {run_para} > lmp.out
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
