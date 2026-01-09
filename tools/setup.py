from jobs.lammpsJob import *
from jobs.alchem import *
from jobs.einstein import *
from jobs.tramp import *
from jobs.sli import *
from jobs.sgmc import *
from tools.logging_config import exapd_logger
from tools.utils import merge_arrays
import os
import numpy as np


def liquidJobs(general, liquid):
    """
    Set up alchemical TI and temperature-ramping jobs for the liquid phase.

    This function reads parameters from the ``general`` and ``liquid`` sections
    of the input JSON file and creates:
    - alchemical thermodynamic-integration jobs (via :class:`alchem`), and
    - temperature-ramping jobs (via :class:`tramp`)
    for a sequence of compositions interpolating between two endpoints.

    Parameters
    ----------
    general : lammpsPara
        General simulation parameters (system, units, project directory, etc.).
    liquid : dict
        Liquid-phase settings.

    Returns
    -------
    list of lammpsJob
        List of all LAMMPS jobs created for the liquid phase.
    """
    liq_dir = f"{general.proj_dir}/liquid"
    if not os.path.isdir(liq_dir):
        try:
            os.mkdir(liq_dir)
        except Exception as e:
            exapd_logger.critical(f"{e}: Cannnot create directory {liq_dir}.")
    data_in = os.path.abspath(liquid["data_in"])
    if not os.path.exists(data_in):
        exapd_logger.critical(f"File {data_in} does not exist.")

    comp0 = liquid["initial_comp"]
    comp1 = liquid["final_comp"]
    comp0 = np.array(comp0) / sum(comp0)
    comp1 = np.array(comp1) / sum(comp1)

    try:
        ncomp = liquid["ncomp"]
    except KeyError:
        ncomp = 10

    try:
        ref_pair_style = liquid["ref_pair_style"]
        ref_pair_coeff = liquid["ref_pair_coeff"]
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
    dn = (n1 - n0) / ncomp

    for icomp in range(ncomp + 1):
        if icomp == 0:
            ref_pair = None
        else:
            ref_pair = gen_ref_pair

        n = n0 + icomp * dn
        n = n.astype(int)

        if sum(n) != natom:
            idx = np.argsort(n)
            if sum(n) > natom:
                n[idx[-1]] -= (sum(n) - natom)
            else:
                n[idx[0]] += (natom - sum(n))

        compdir = f"{liq_dir}/comp{icomp}"
        if not os.path.isdir(compdir):
            try:
                os.mkdir(compdir)
            except Exception as e:
                exapd_logger.critical(
                    f"{e}: Cannot create directory {compdir}!")

        if ref_pair is None:
            pre_job_dir = f"{compdir}/tramp/T{Tlist[-1]:g}"
            pre_var_names = ["vol"]
            pre_var_values = ["Volume"]
            depend = (pre_job_dir, pre_var_names, pre_var_values)
        else:
            depend = None

        liq_alchem = alchem(
            data_in, dlbd, Tmax,
            f"{compdir}/alchem", ref_pair, n
        )
        liq_alchem.setup(general, depend)
        liq_jobs += liq_alchem.get_joblist()

        liq_tramp = tramp(data_in, Tlist, f"{compdir}/tramp", n)
        liq_tramp.setup(general)
        if depend:
            liq_tramp.get_joblist()[-1]._priority = 0
        liq_jobs += liq_tramp.get_joblist()

    return liq_jobs


def solidJobs(general, solid):
    """
    Set up Einstein-crystal TI and temperature-ramping jobs for solid phases.

    Parameters
    ----------
    general : lammpsPara
        General simulation parameters.
    solid : dict
        Solid-phase settings.

    Returns
    -------
    list of lammpsJob
        List of all LAMMPS jobs created for solid phases.
    """
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

        sol_pre = tramp(
            data_in, Tlist[:1], f"{phdir}/tramp", barostat=barostat)
        sol_pre.setup(general, boxdims=True, msd=True)
        for job in sol_pre.get_joblist():
            job._priority = 0
            sol_jobs.append(job)

        sol_tramp = tramp(
            data_in, Tlist[1:], f"{phdir}/tramp", barostat=barostat)
        sol_tramp.setup(general)
        sol_jobs += sol_tramp.get_joblist()

        natom, ntyp, nab = read_lmp_data(data_in, read_nab=True)
        pre_job_dir = f"{phdir}/tramp/T{Tlist[0]:g}"
        pre_var_names = "fxlo fxhi fylo fyhi fzlo fzhi".split()
        pre_var_values = "Xlo Xhi Ylo Yhi Zlo Zhi".split()

        if barostat == "tri":
            pre_var_names += "fxy fxz fyz".split()
            pre_var_values += "Xy Xz Yz".split()

        for i in range(ntyp):
            if nab[i] > 0:
                pre_var_names.append(f"msd{i + 1}")
                pre_var_values.append(f"c_c{i + 1}[4]")

        depend = (pre_job_dir, pre_var_names, pre_var_values)
        sol_ti = einstein(
            data_in, dlbd, Tlist[0], directory=f"{phdir}/einstein")
        sol_ti.setup(general, barostat, depend)
        sol_jobs += sol_ti.get_joblist()

    return sol_jobs


def sliJobs(general, sli):
    """
    Set up solid–liquid interface (SLI) simulations.

    Parameters
    ----------
    general : lammpsPara
        General simulation parameters.
    sli : dict
        SLI settings.

    Returns
    -------
    list of lammpsJob
        List of all LAMMPS jobs created for SLI simulations.
    """
    sli_dir = f"{general.proj_dir}/sli"
    if not os.path.isdir(sli_dir):
        try:
            os.mkdir(sli_dir)
        except Exception as e:
            exapd_logger.critical(f"{e}: Cannnot create directory {sli_dir}")

    phases = sli["phases"]

    try:
        Tlist = np.sort(sli["Tlist"])
    except KeyError:
        Tlist = None

    try:
        Tmin = sli["Tmin"]
        Tmax = sli["Tmax"]
        dT = sli["dT"]
        if Tlist is None:
            Tlist = np.arange(Tmin, Tmax + 0.1 * dT, dT)
        else:
            Tlist = merge_arrays(Tlist, np.arange(Tmin, Tmax + 0.1 * dT, dT))
    except KeyError:
        if Tlist is None:
            exapd_logger.critical("Tlist cannot be created for solid.")

    try:
        Tmelt = sli["Tmelt"]
    except Exception as e:
        exapd_logger.critical(
            f"{e}: a high T for melting half of the box is needed.")

    try:
        ntarget = sli["ntarget"]
    except KeyError:
        ntarget = 5000

    try:
        replicate = sli["replicate"]
    except KeyError:
        replicate = 2

    try:
        orient = sli["orientation"]
    except KeyError:
        orient = "z"

    if orient not in ("x", "y", "z"):
        exapd_logger.critical(
            "Error: orientation needs to be x or y or z for SLI.")

    sli_jobs = []

    for ph in phases:
        ph_file = os.path.abspath(ph)
        if not os.path.exists(ph_file):
            exapd_logger.critical(f"File {ph_file} does not exist.")

        name, form = ph_file.split('/')[-1].split('.')
        phdir = f"{sli_dir}/{name}"
        if not os.path.isdir(phdir):
            try:
                os.mkdir(phdir)
            except Exception as e:
                exapd_logger.critical(f"{e}: Cannot create directory {phdir}.")

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

        mysli = sli_simulator(
            data_in, Tlist, Tmelt, f"{phdir}",
            replicate, orient, barostat
        )
        mysli.setup(general)
        sli_jobs += mysli.get_joblist()

    return sli_jobs


def sgmcJobs(general, sgmc):
    """
    Set up semi-grand canonical Monte Carlo (SGMC) jobs.

    Parameters
    ----------
    general : lammpsPara
        General simulation parameters.
    sgmc : dict
        SGMC settings.

    Returns
    -------
    list of lammpsJob
        List of all LAMMPS jobs created for SGMC simulations.
    """
    sgmc_dir = f"{general.proj_dir}/sgmc"
    if not os.path.isdir(sgmc_dir):
        try:
            os.mkdir(sgmc_dir)
        except Exception as e:
            exapd_logger.critical(f"{e}: Cannnot create directory {sgmc_dir}")

    phases = sgmc["phases"]

    try:
        Tlist = np.sort(sgmc["Tlist"])
    except KeyError:
        Tlist = None

    try:
        Tmin = sgmc["Tmin"]
        Tmax = sgmc["Tmax"]
        dT = sgmc["dT"]
        if Tlist is None:
            Tlist = np.arange(Tmin, Tmax + 0.1 * dT, dT)
        else:
            Tlist = merge_arrays(Tlist, np.arange(Tmin, Tmax + 0.1 * dT, dT))
    except KeyError:
        if Tlist is None:
            exapd_logger.critical("Tlist cannot be created for solid.")

    try:
        mu_list = np.sort(sgmc["mu_list"])
    except KeyError:
        mu_list = None

    try:
        mu_min = sgmc["mu_min"]
        mu_max = sgmc["mu_max"]
        dmu = sgmc["dmu"]
        if mu_list is None:
            mu_list = np.arange(mu_min, mu_max + 0.1 * dmu, dmu)
        else:
            mu_list = merge_arrays(
                mu_list,
                np.arange(mu_min, mu_max + 0.1 * dmu, dmu)
            )
    except KeyError:
        if mu_list is None:
            exapd_logger.critical("mu_list cannot be created for solid.")

    try:
        ntarget = sgmc["ntarget"]
    except KeyError:
        ntarget = 5000

    sgmc_jobs = []

    for ph in phases:
        ph_file = os.path.abspath(ph)
        if not os.path.exists(ph_file):
            exapd_logger.critical(f"File {ph_file} does not exist.")

        name, form = ph_file.split('/')[-1].split('.')
        phdir = f"{sgmc_dir}/{name}"
        if not os.path.isdir(phdir):
            try:
                os.mkdir(phdir)
            except Exception as e:
                exapd_logger.critical(f"{e}: Cannot create directory {phdir}.")

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

        mysgmc = sgmc_simulator(
            data_in, Tlist, mu_list, f"{phdir}", barostat
        )
        mysgmc.setup(general)
        sgmc_jobs += mysgmc.get_joblist()

    return sgmc_jobs
