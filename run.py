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
import sys

if __name__ == '__main__':
    inp = ConfigManager()
    run_config = inp["run"]

    load_parsl_config(run_config)
    exapd_logger.configure("INFO")

    # set up general parameters
    try:
        general = lammpsPara(inp["general"])
    except Exception as e:
        exapd_logger.critical(f"{e}: Setting up lammpsPara failed")

    # set up liquid jobs
    try:
        liq_jobs = liquidJobs(general, inp["liquid"])
    except Exception as e:
        # exapd_logger.critical(f"{e}: Setting up liquid jobs failed")
        liq_jobs = []
        exapd_logger.warning(f"{e}: Setting up liquid jobs failed")

    # set up solid jobs
    try:
        sol_jobs = solidJobs(general, inp["solid"])
    except Exception as e:
        exapd_logger.critical(f"{e}: Setting up solid jobs failed")

    jobs = liq_jobs + sol_jobs
    print('Number of MD jobs', len(jobs))

    # launch jobs
    jobs.sort(key=lambda job: job._priority)
    parsl_job_dict = {}
    lmp_cpu_exe = run_config["cpu_exe"]
    lmp_gpu_exe = run_config["gpu_exe"]
    for job in jobs:
        if job._depend:
            dep_future = parsl_job_dict[job._depend[0]]
        else:
            dep_future = None
        if job._arch == "cpu":
            try:
                parsl_job_dict[job._dir] = cpu_lammps(job._dir, job._script,
                                                      lmp_cpu_exe,
                                                      dep_future=dep_future,
                                                      depend=job._depend)
            except Exception as e:
                exapd_logger.warning(
                    f"{e}: Launching cpu job failed: {job._dir}")
        if job._arch == "gpu":
            try:
                parsl_job_dict[job._dir] = gpu_lammps(job._dir, job._script,
                                                      lmp_gpu_exe,
                                                      dep_future=dep_future,
                                                      depend=job._depend)
            except Exception as e:
                exapd_logger.warning(
                    f"{e}: Launching gpu job failed: {job._dir}")

    wait_for_current_tasks()
    print("all done")
    parsl.dfk().cleanup()
