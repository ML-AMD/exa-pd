"""
Main execution script for ExaPD LAMMPS workflows.

This script is the primary entry point for running ExaPD simulations.
It performs the following tasks:

1. Parse the JSON configuration file using :class:`ConfigManager`.
2. Load and initialize the Parsl execution configuration.
3. Set up LAMMPS simulation parameters.
4. Generate job workflows (liquid, solid, etc.).
5. Submit jobs to CPU or GPU backends using Parsl.
6. Wait for completion and perform cleanup.

The script is intended to be executed directly from the command line.

Example
-------
Run the workflow with a configuration file::

    python run.py --config input.json
"""

from jobs.lammpsJob import *
from tools.setup import *
import json
import parsl
from parsl import wait_for_current_tasks
from parsl_configs.perlmutter import PerlmutterConfig
from parsl_tasks.lammps import cpu_lammps, gpu_lammps
from tools.logging_config import exapd_logger
from tools.config_manager import ConfigManager
from parsl_configs.config_registry import load_parsl_config
import os


def main():    
    """
    Execute the ExaPD workflow.

    This block orchestrates the full simulation lifecycle:
    configuration parsing, job creation, submission, monitoring,
    and final cleanup.
    """

    # read configuration
    inp = ConfigManager()
    run_config = inp["run"]

    # initialize Parsl
    load_parsl_config(run_config)
    exapd_logger.configure("INFO")

    # set up general LAMMPS parameters
    try:
        general = lammpsPara(inp["general"])
    except Exception as e:
        exapd_logger.critical(f"{e}: Setting up lammpsPara failed")

    # initialize job list
    jobs = []

    # set up liquid jobs
    if "liquid" in inp:
        try:
            jobs += liquidJobs(general, inp["liquid"])
        except Exception as e:
            exapd_logger.critical(f"{e}: Setting up liquid jobs failed")

    # set up solid jobs
    if "solid" in inp:
        try:
            jobs += solidJobs(general, inp["solid"])
        except Exception as e:
            exapd_logger.critical(f"{e}: Setting up solid jobs failed")

    # sort and launch jobs
    jobs.sort(key=lambda job: job._priority)
    parsl_job_dict = {}

    lmp_cpu_exe = run_config["cpu_exe"]
    lmp_gpu_exe = run_config["gpu_exe"]

    for job in jobs:
        if os.path.exists(f"{job._dir}/DONE"):
            continue

        exapd_logger.info(
            f"Submitted {job._dir}, {job._arch}, priority: {job._priority}"
        )

        if job._depend and job._depend[0] in parsl_job_dict:
            dep_future = parsl_job_dict[job._depend[0]]
        else:
            dep_future = None

        if job._arch == "cpu":
            try:
                parsl_job_dict[job._dir] = cpu_lammps(
                    job._dir,
                    job._script,
                    lmp_cpu_exe,
                    dep_future=dep_future,
                    depend=job._depend
                )
            except Exception as e:
                exapd_logger.warning(
                    f"{e}: Launching cpu job failed: {job._dir}"
                )

        if job._arch == "gpu":
            try:
                parsl_job_dict[job._dir] = gpu_lammps(
                    job._dir,
                    job._script,
                    lmp_gpu_exe,
                    dep_future=dep_future,
                    depend=job._depend
                )
            except Exception as e:
                exapd_logger.warning(
                    f"{e}: Launching gpu job failed: {job._dir}"
                )

    # wait for all jobs to finish
    wait_for_current_tasks()
    print("all done")

    # clean up Parsl
    parsl.dfk().cleanup()

if __name__ == '__main__':
    main()    
