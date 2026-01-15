"""
Parsl task wrappers for launching LAMMPS runs.

This module defines Parsl ``bash_app`` tasks for running LAMMPS on
GPU and CPU executors. The tasks:

- optionally preprocess dependency outputs into ``-v`` command-line variables,
- run LAMMPS in a specified working directory, and
- create a ``DONE`` marker file on success.

The functions return bash scripts as strings (as required by Parsl ``bash_app``).

Notes
-----
- ``gpu_lammps`` runs LAMMPS directly using the provided executable.
- ``cpu_lammps`` uses ``srun`` and ``shifter`` (NERSC-style environment) to launch
  the CPU job.
"""

import parsl
from parsl import python_app, bash_app


@bash_app(executors=["gpu"])
def gpu_lammps(directory, script, lmp_exe,
               dep_future=None, depend=None,
               stdout=parsl.AUTO_LOGNAME, stderr=parsl.AUTO_LOGNAME):
    """
    Run a LAMMPS job on the Parsl GPU executor.

    Parameters
    ----------
    directory : str
        Working directory where the job is executed.
    script : str
        Path to the LAMMPS input script file.
    lmp_exe : str
        LAMMPS executable command.
    dep_future : parsl.dataflow.futures.AppFuture, optional
        Dependency future to enforce execution ordering in Parsl.
        (Not explicitly used in the function body, but included so Parsl
        will enforce the dependency.)
    depend : object, optional
        Dependency specification passed to :func:`jobs.lammpsJob.pre_process`.
        If provided, pre-processing yields additional ``-v`` variables
        passed to LAMMPS.
    stdout : str, optional
        Parsl stdout log destination (default: ``parsl.AUTO_LOGNAME``).
    stderr : str, optional
        Parsl stderr log destination (default: ``parsl.AUTO_LOGNAME``).

    Returns
    -------
    str
        Bash script that runs LAMMPS and writes ``DONE`` on success.
    """
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
    """
    Run a LAMMPS job on the Parsl CPU executor using srun + shifter.

    Parameters
    ----------
    directory : str
        Working directory where the job is executed.
    script : str
        Path to the LAMMPS input script file.
    lmp_exe : str
        LAMMPS executable command.
    ncpu : int, optional
        Number of MPI ranks to request via ``srun`` (default: 32).
    dep_future : parsl.dataflow.futures.AppFuture, optional
        Dependency future to enforce execution ordering in Parsl.
        (Not explicitly used in the function body, but included so Parsl
        will enforce the dependency.)
    depend : object, optional
        Dependency specification passed to :func:`jobs.lammpsJob.pre_process`.
        If provided, pre-processing yields additional ``-v`` variables
        passed to LAMMPS.
    stdout : str, optional
        Parsl stdout log destination (default: ``parsl.AUTO_LOGNAME``).
    stderr : str, optional
        Parsl stderr log destination (default: ``parsl.AUTO_LOGNAME``).

    Returns
    -------
    str
        Bash script that launches LAMMPS using ``srun`` and ``shifter`` and
        writes ``DONE`` on success.

    Notes
    -----
    The current command hardcodes a Shifter image:

    ``nersc/lammps_all:23.08``

    Adjust this string if your environment uses a different container image.
    """
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
