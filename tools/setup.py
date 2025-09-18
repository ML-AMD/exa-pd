from jobs.lammpsJob import *
from jobs.alchem import *
from jobs.einstein import *
from jobs.tramp import *
from tools.logging_config import exapd_logger
from tools.utils import merge_arrays
import os
import sys


def liquidJobs(general, liquid):
    '''
    set up TI and T-ramping jobs for liquid, taking input from the general
    and liquid sections from the input JSon file.
    '''
    # read input
<<<<<<< HEAD
    try:
        mode = liquid["mode"]
    except BaseException:
        mode = "scratch"
    if mode not in ["scratch", "restart"]:
        return []  # do nothing
    liq_dir = f"{general.proj_dir}/liquid"
    if not os.path.isdir(liq_dir):
        if mode == "scratch":
            try:
                os.mkdir(liq_dir)
            except BaseException:
                print(f"Error: cannnot create directory {liq_dir}")
                sys.exit(1)
        else:
            raise Exception(f"{liq_dir} does not exist for restarting job!")
    data_in = os.path.abspath(liquid["data_in"])
    if not os.path.exists(data_in):
        raise Exception(f"Error file {data_in} doesn't exist!")
=======
    liq_dir = f"{general.proj_dir}/liquid"
    if not os.path.isdir(liq_dir):
        try:
            os.mkdir(liq_dir)
        except Exception as e:
            exapd_logger.critical(f"{e}: Cannnot create directory {liq_dir}.")
    data_in = os.path.abspath(liquid["data_in"])
    if not os.path.exists(data_in):
        exapd_logger.critical(f"File {data_in} does not exist.")
>>>>>>> origin/fz_refactor

    comp0 = liquid["initial_comp"]
    comp1 = liquid["final_comp"]
    # normalize comp0 and comp1
    comp0 = np.array(comp0) / sum(comp0)
    comp1 = np.array(comp1) / sum(comp1)
    try:
        ncomp = liquid["ncomp"]
<<<<<<< HEAD
    except BaseException:
=======
    except KeyError:
>>>>>>> origin/fz_refactor
        ncomp = 10
    try:
        ref_pair_style = liquid["ref_pair_style"]
        ref_pair_coeff = liquid["ref_pair_coeff"]
<<<<<<< HEAD
        ref_pair = lammpsPair(ref_pair_style, ref_pair_coeff)
    except BaseException:
        ref_pair = None
    try:
        dlbd = liquid["dlbd"]
    except BaseException:
        dlbd = 0.05
    Tmin = liquid["Tmin"]
    Tmax = liquid["Tmax"]
    dT = liquid["dT"]
    Tlist = np.arange(Tmin, Tmax + 0.1 * dT, dT)
    liq_jobs = []
    natom, ntyp = read_lmp_data(data_in)
    n0 = natom * np.asarray(comp0)
    n1 = natom * np.asarray(comp1)
=======
        gen_ref_pair = lammpsPair(ref_pair_style, ref_pair_coeff)
    except KeyError:
        gen_ref_pair = None
    try:
        dlbd = liquid["dlbd"]
    except KeyError:
        dlbd = 0.05
    try:
        Tlist = np.sort(liquid["Tlist"])
    except KeyError:
        Tlist = None
    try:
        Tmin = liquid["Tmin"]
        Tmax = liquid["Tmax"]
        dT = liquid["dT"]
        if Tlist is None:
            Tlist = np.arange(Tmin, Tmax + 0.1 * dT, dT)
        else:
            Tlist = merge_arrays(Tlist, np.arange(Tmin, Tmax + 0.1 * dT, dT))
    except KeyError:
        if Tlist is None:
            exapd_logger.critical("Tlist cannot be created for liquid.")
    liq_jobs = []
    natom, ntyp = read_lmp_data(data_in)
    n0 = natom * comp0
    n1 = natom * comp1
>>>>>>> origin/fz_refactor
    dn = (n1 - n0) / ncomp
    for icomp in range(ncomp + 1):
        if icomp == 0:
            ref_pair = None  # always use UFM as ref for comp0
        else:
            ref_pair = gen_ref_pair
        n = n0 + icomp * dn
        n = n.astype(int)
        # fix possible rounding error
        if sum(n) != natom:
            idx = np.argsort(n)
            if sum(n) > natom:
                n[idx[-1]] -= (sum(n) - natom)
            else:
                n[idx[0]] += (natom - sum(n))
        compdir = f"{liq_dir}/comp{icomp}"
        if not os.path.isdir(compdir):
<<<<<<< HEAD
            if mode == "scratch":
                try:
                    os.mkdir(compdir)
                except BaseException:
                    print(f"Error: cannot create directory {compdir}!")
                    sys.exit(1)
            else:
                raise Exception(f"{compdir} does not exist for restarting job!")
        # set up alchem jobs
        liq_alchem = alchem(data_in, dlbd, Tmax, f"{compdir}/alchem", mode, ref_pair, n)
        res = liq_alchem.setup(general)
=======
            try:
                os.mkdir(compdir)
            except Exception as e:
                exapd_logger.critical(
                    f"{e}: Cannot create directory {compdir}!")
        # set up alchem jobs
        if ref_pair is None:  # set up depend feature for UFM
            pre_job_dir = f"{compdir}/tramp/T{Tlist[-1]:g}"
            pre_var_names = ["vol"]
            pre_var_values = ["Volume"]
            depend = (pre_job_dir, pre_var_names, pre_var_values)
        else:
            depend = None
        liq_alchem = alchem(data_in, dlbd, Tmax,
                            f"{compdir}/alchem", ref_pair, n)
        res = liq_alchem.setup(general, depend)
>>>>>>> origin/fz_refactor
        liq_jobs += liq_alchem.get_joblist()
        # set up T-ramping jobs
        liq_tramp = tramp(data_in, Tlist, f"{compdir}/tramp", n)
        res = liq_tramp.setup(general)
        if depend:
            liq_tramp.get_joblist()[-1]._priority = 0
        liq_jobs += liq_tramp.get_joblist()
    return liq_jobs


def solidJobs(general, solid):
    '''
    set up TI and T-ramping jobs for solid, taking input from the general
    and solid sections from the input JSon file.
    '''
    # read input
    sol_dir = f"{general.proj_dir}/solid"
    if not os.path.isdir(sol_dir):
        try:
            os.mkdir(sol_dir)
        except Exception as e:
            exapd_logger.critical(f"{e}: Cannnot create directory {sol_dir}")

    phases = solid["phases"]
    try:
        dlbd = solid["dlbd"]
    except KeyError:
        dlbd = 0.05
    try:
        Tlist = np.sort(solid["Tlist"])
    except KeyError:
        Tlist = None
    try:
        Tmin = solid["Tmin"]
        Tmax = solid["Tmax"]
        dT = solid["dT"]
        if Tlist is None:
            Tlist = np.arange(Tmin, Tmax + 0.1 * dT, dT)
        else:
            Tlist = merge_arrays(Tlist, np.arange(Tmin, Tmax + 0.1 * dT, dT))
    except KeyError:
        if Tlist is None:
            exapd_logger.critical("Tlist cannot be created for solid.")
    try:
        ntarget = solid["ntarget"]
    except KeyError:
        ntarget = 5000
    sol_jobs = []
    for ph in phases:
        # supports lammps format or other formats that can be converted
        # to lammps format by ASE.
        ph_file = os.path.abspath(ph)
        if not os.path.exists(ph_file):
            exapd_logger.critical(f"File {ph_file} does not exist.")
        name, form = ph_file.split('/')[-1].split('.')
        phdir = f"{sol_dir}/{name}"
        if not os.path.isdir(phdir):
            try:
                os.mkdir(phdir)
            except Exception as e:
                exapd_logger.critical(f"{e}: Cannot create directory {phdir}.")
        # set up Frankel-Ladd jobs
        if form == "lammps":
            data_in = ph_file
            barostat = get_lammps_barostat(data_in)
        else:
            data_in = f"{phdir}/{name}.lammps"
            try:
                barostat = create_lammps_supercell(
                    general.system, ph_file, data_in, ntarget=ntarget)
            except Exception as e:
                exapd_logger.critical(
                    f"{e}: ASE could not generate the lammps input for {ph_file}.")
        # set up pre jobs
        sol_pre = tramp(data_in, Tlist[:1], f"{phdir}/tramp",
                        barostat=barostat)
        res = sol_pre.setup(general, boxdims=True, msd=True)
        for job in sol_pre.get_joblist():
            job._priority = 0  # set to highest priority
            sol_jobs.append(job)
        # set up other T-ramping jobs
        sol_tramp = tramp(data_in, Tlist[1:], f"{phdir}/tramp",
                          barostat=barostat)
        res = sol_tramp.setup(general)
        sol_jobs += sol_tramp.get_joblist()
        # set up einstein jobs
        natom, ntyp, nab = read_lmp_data(data_in, read_nab=True)
        pre_job_dir = f"{phdir}/tramp/T{Tlist[0]:g}"
        pre_var_names = "fxlo fxhi fylo fyhi fzlo fzhi".split()
        pre_var_values = "Xlo Xhi Ylo Yhi Zlo Zhi".split()
        if barostat == "tri":
            pre_var_names += "fxy fxz fyz".split()
            pre_var_values += "Xy Xz Yz".split()
        for i in range(ntyp):  # msd
            if nab[i] > 0:
                pre_var_names.append(f"msd{i + 1}")
                pre_var_values.append(f"c_c{i + 1}[4]")
        depend = (pre_job_dir, pre_var_names, pre_var_values)
        sol_ti = einstein(data_in, dlbd, Tlist[0],
                          directory=f"{phdir}/einstein")
        res = sol_ti.setup(general, barostat, depend)
        sol_jobs += sol_ti.get_joblist()
    return sol_jobs
