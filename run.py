from lammpsJob import *
from setup import *
import json

inp = json.load(open("input.json", "r"))
# set up general parameters 
general = lammpsPara(inp["general"])

# set up liquid jobs
#liq_jobs = liquidJobs(general, inp["liquid"])

# set up solid jobs
sol_jobs = solidJobs(general, inp["solid"])

#jobs = liq_jobs + sol_jobs
jobs = sol_jobs
print(len(jobs))
# launch jobs
### set up parsl
import parsl
from parsl import wait_for_current_tasks
from parslTools import PerlmutterConfig
from parslTools import cpu_lammps, gpu_lammps

parsl.load(PerlmutterConfig())

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
for job_dir in pre_dict: print(job_dir)
print("reg_job",len(reg_dict))
for job_dir in reg_dict: print(job_dir)
print("dep_job",len(dep_dict))
for job_dir in dep_dict: print(job_dir)

parsl_job_dict={}
lmp_exe = "/global/homes/f/fzhang/install/lammps/src/lmp_gpu_serial"
for job_dir,job in pre_dict.items():
    if job._arch == "cpu":
        parsl_job_dict[job_dir] = cpu_lammps(job._dir,job._script)
    if job._arch == "gpu":
        parsl_job_dict[job_dir] = gpu_lammps(job._dir,job._script,lmp_exe)

for job_dir,job in reg_dict.items():
    if job._arch == "cpu":
        parsl_job_dict[job_dir] = cpu_lammps(job._dir,job._script)
    if job._arch == "gpu":
        parsl_job_dict[job_dir] = gpu_lammps(job._dir,job._script,lmp_exe)

from lammpsJob import pre_process
for job_dir,job in dep_dict.items():
    parsl_job_dict[job._depend[0]].result()
    run_para = pre_process(job._depend)
    if job._arch == "cpu":
        parsl_job_dict[job_dir] = cpu_lammps(job._dir,job._script,run_para=run_para)
    if job._arch == "gpu":
        parsl_job_dict[job_dir] = gpu_lammps(job._dir,job._script,lmp_exe,run_para=run_para)


wait_for_current_tasks()
print("all done")
parsl.dfk().cleanup()

        



