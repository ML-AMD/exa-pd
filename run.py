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
        exapd_logger.critical(f"{e}: Setting up liquid jobs failed")    

    # set up solid jobs
    try:
        sol_jobs = solidJobs(general, inp["solid"])
    except Exception as e:
        exapd_logger.critical(f"{e}: Setting up solid jobs failed")    

    jobs = liq_jobs + sol_jobs
    print('Number of MD jobs',len(jobs))

    # launch jobs
    lmp_jobs_dict={}
    for job in jobs:
        lmp_jobs_dict[job._dir]=job

    for job_dir,job in lmp_jobs_dict.items():
        if job._depend is not None:
            print(job_dir, job._arch, job._depend[0])
        else:
            print(job_dir, job._arch, job._depend)

    pre_dict = {}
    reg_dict = {}
    dep_dict = {}
    for job_dir,job in lmp_jobs_dict.items():
        reg_dict[job_dir] = job
        if job._depend != None:
            dep_dict[job._dir] = job
            if not job._depend[0] in pre_dict:
                pre_dict[job._depend[0]] = lmp_jobs_dict[job._depend[0]]

    for job_dir in pre_dict:
        if job_dir in reg_dict:
            del reg_dict[job_dir]

    for job_dir in dep_dict:
        if job_dir in dep_dict:
            del reg_dict[job_dir]

    print("pre_job",len(pre_dict))
    print("reg_job",len(reg_dict))
    print("dep_job",len(dep_dict))

    parsl_job_dict={}
    lmp_cpu_exe = run_config["cpu_exe"]
    lmp_gpu_exe = run_config["gpu_exe"]
    for job_dir,job in pre_dict.items():
        if job._arch == "cpu":
            try:
                parsl_job_dict[job_dir] = cpu_lammps(job._dir,job._script,lmp_cpu_exe)
            except Exception as e:
                exapd_logger.warning(f"{e}: Launching cpu job failed: {job_dir}") 
        if job._arch == "gpu":
            try:
                parsl_job_dict[job_dir] = gpu_lammps(job._dir,job._script,lmp_gpu_exe)
            except Exception as e:
                exapd_logger.warning(f"{e}: Launching gpu job failed: {job_dir}")

    for job_dir,job in reg_dict.items():
        if job._arch == "cpu":
            try:
                parsl_job_dict[job_dir] = cpu_lammps(job._dir,job._script,lmp_cpu_exe)
            except Exception as e:
                exapd_logger.warning(f"{e}: Launching cpu job failed: {job_dir}")
        if job._arch == "gpu":
            try:
                parsl_job_dict[job_dir] = gpu_lammps(job._dir,job._script,lmp_gpu_exe)
            except Exception as e:
                exapd_logger.warning(f"{e}: Launching gpu job failed: {job_dir}")

    for job_dir,job in dep_dict.items():
        if job._arch == "cpu":
            try:
                parsl_job_dict[job_dir] = cpu_lammps(job._dir,job._script,\
                    lmp_cpu_exe, dep_future=parsl_job_dict[job._depend[0]], depend=job._depend)
            except Exception as e:
                exapd_logger.warning(f"{e}: Launching cpu job failed: {job_dir}")
        if job._arch == "gpu":
            try:
                parsl_job_dict[job_dir] = gpu_lammps(job._dir,job._script,\
                    lmp_gpu_exe, dep_future=parsl_job_dict[job._depend[0]], depend=job._depend)
            except Exception as e:
                exapd_logger.warning(f"{e}: Launching gpu job failed: {job_dir}")


    wait_for_current_tasks()
    print("all done")
    parsl.dfk().cleanup()

