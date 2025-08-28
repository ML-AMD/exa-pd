import parsl
from parsl import python_app, bash_app

@bash_app(executors=["gpu"])
def gpu_lammps(directory, script, lmp_exe, run_para="", dep_future = None, stdout=parsl.AUTO_LOGNAME, stderr=parsl.AUTO_LOGNAME):
    return f"""
    dir="{directory}"
    cd {directory}
    echo "Running command: {lmp_exe} -sf gpu -pk gpu 1 -in {script} {run_para} > lmp.out"
    {lmp_exe} -in {script} {run_para} > lmp.out
    touch DONE
    exapd_logger.info(f"Done job: {directory}")
    """

@bash_app(executors=["cpu"])
def cpu_lammps(directory, script, lmp_exe, run_para="", ncpu=32, dep_future = None, depend = None, stdout=parsl.AUTO_LOGNAME, stderr=parsl.AUTO_LOGNAME):
    if not run_para:
        from ..lammpsJob import pre_process
        run_para = pre_process(depend)

    return f"""
    dir="{directory}"
    cd {directory}
    echo "Running command: srun -N 1 -n {ncpu} -c 2 --cpu-bind=cores --exact shifter --image=nersc/lammps_lite:23.08 {lmp_exe} -in {script} {run_para} > lmp.out"
    srun -N 1 -n {ncpu} -c 2 --cpu-bind=cores --exact shifter --image=nersc/lammps_lite:23.08 {lmp_exe} -in {script} {run_para} > lmp.out
    touch DONE
    exapd_logger.info(f"Done job: {directory}")
    """
