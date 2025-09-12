import parsl
from parsl import python_app, bash_app


@bash_app(executors=["gpu"])
def gpu_lammps(directory, script, lmp_exe,
               dep_future=None, depend=None,
               stdout=parsl.AUTO_LOGNAME, stderr=parsl.AUTO_LOGNAME):
    run_para = ""
    if depend:
        from jobs.lammpsJob import pre_process
        run_para = pre_process(depend) or ""

    return f"""\
    set -e
    cd "{directory}"
    echo "Running command: {lmp_exe} -in {script} {run_para} > lmp.out"
    {lmp_exe} -in "{script}" {run_para} > lmp.out && touch DONE
    """


@bash_app(executors=["cpu"])
def cpu_lammps(directory, script, lmp_exe, ncpu=32,
               dep_future=None, depend=None,
               stdout=parsl.AUTO_LOGNAME, stderr=parsl.AUTO_LOGNAME):
    run_para = ""
    if depend:
        from jobs.lammpsJob import pre_process
        run_para = pre_process(depend) or ""

    return f"""\
    set -e
    cd "{directory}"
    echo "Running command: srun -N 1 -n {ncpu} -c 2 --cpu-bind=cores --exact \\
    shifter --image=nersc/lammps_all:23.08 {lmp_exe} -in {script} {run_para} > lmp.out"
    srun -N 1 -n {ncpu} -c 2 --cpu-bind=cores --exact \\
    shifter --image=nersc/lammps_all:23.08 {lmp_exe} -in "{script}" {run_para} > lmp.out && touch DONE
    """
