from jobs.lammpsJob import *
from jobs.alchem import *
from jobs.einstein import *
from jobs.tramp import *
import matplotlib.pyplot as plt
from tools.logging_config import exapd_logger
from tools.utils import merge_arrays
import os


def process_liquid(general, liquid, write_file=True):
    '''
    post-process liquid calculations.
    G0: Gibbs free energy of the pure phase at T = T_alchem (Tlist[-1])
    '''
    liq_dir = f"{general.proj_dir}/liquid"
    if not os.path.isdir(liq_dir):
        rexapd_logger.critical(
            f"{liq_dir} does not exist for post-processing!")
    data_in = os.path.abspath(liquid["data_in"])
    comp0 = liquid["initial_comp"]
    comp1 = liquid["final_comp"]
    # normalize comp0 and comp1
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
    # create a finer T-mesh for smooth free energy
    if general.units == "lj":
        kb = 1
        ddT = 0.01
    else:
        kb = 8.617333262e-5
        ddT = 10
    R = 8.3144598 # gas constant
    Tall = np.arange(min(Tlist), max(Tlist) + 0.1 * ddT, ddT)
    tdb = ''  # entry of the liquid phase in the TDB file
    natom, ntyp = read_lmp_data(data_in)
    n0 = natom * np.asarray(comp0)
    n1 = natom * np.asarray(comp1)
    dn = (n1 - n0) / ncomp
    xall = np.zeros(ncomp + 1)
    for i in range(len(comp0)):
        if comp0[i] < comp1[i]:
            comp_idx = i
            break
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
            rexapd_logger.critical(f"{compdir} does not exist for post-processing.")
        #  process alchem jobs
        liq_alchem = alchem(data_in, dlbd, Tmax,
                            f"{compdir}/alchem", ref_pair=ref_pair, nab=n)
        det_G = liq_alchem.process(general)
        # process T-ramping jobs
        liq_tramp = tramp(data_in, Tlist, f"{compdir}/tramp", nab=n)
        res = liq_tramp.process()
        if general.units == "lj":
            dT = 0.01
        else:
            dT = 10
        arrT, arrH = fix_enthalpy(res[:, 0], res[:, 1], phase="liquid")
        if icomp == 0:
            G = Gibbs_Helmholtz(arrT, arrH, Tlist[-1], det_G, Tall)
            Gall = G.reshape(-1, 1)
        else:
            if ref_pair is None:
                G0 = det_G # det_G is absolute for using UFM as ref
            else:
                G0 = det_G + Gall[-1, 0] # det_G is relative G to G(x=0) otherwise 
            G = Gibbs_Helmholtz(arrT, arrH, Tlist[-1], G0, Tall)
            Gall = np.column_stack((Gall, G))
        xall[icomp] = n[comp_idx] / natom
    if write_file:
        header = f"Gibbs free energy of the liquid phase\n   T  x_{general.system[comp_idx]} = "
        for x in xall:
            header += f"{x:.6f} "
        np.savetxt(f"{general.proj_dir}/g.liq.dat",
                   np.column_stack((Tall, Gall)), fmt="%.6f", header=header)

    # generate TDB entry for (sub)binary system
    # check if comp0 and comp1 belong to a sub-binary system 
    create_tdb = True
    binary = [None, None] 
    for i in range(len(comp0)):
        if comp0[i] > comp1[i]:
            if binary[0] is None:
                binary[0] = general.system[i]
            else:
                create_tdb = False
                break
        elif comp0[i] < comp1[i]:
            if binary[1] is None:
                binary[1] = general.system[i]
            else:
                create_tdb = False
                break
        else:
            if comp0[i] != 0:
                create_tdb = False
                break
    if not create_tdb:
        exapd_logger.info("TDB only generated for (sub)binary systems.")
        return ""
    # set end members, extrapolate if necessary
    if xall[0] == 0:
        G0 = Gall[:, 0]
    else:
        G0 = np.zeros(len(Tall))
        for i in range(len(G0)):
            G0[i] = scipy.interpolate.interp1d(xall, Gall[i],
                    kind='linear', fill_value='extrapolate')(0)
    if xall[-1] == 1:
        G1 = Gall[:, -1]
    else:
        G1 = np.zeros(len(Tall))
        for i in range(len(G1)):
            G1[i] = scipy.interpolate.interp1d(xall, Gall[i],
                    kind='linear', fill_value='extrapolate')(1)
            
    phase, end, rk = create_tdb_header_binsol("liq", binary[:2])
    tdb += phase # phase definition in TDB
    # create TDB entry for end member x = 0
    res = scipy.optimize.curve_fit(tlogpoly, Tall, G0, np.ones(6))
    params = res[0] * R / kb  # default unit in CALPHAD is J/mol
    tdb += f"{end[0]} {Tlist[0]:g} {params[0]:+.12g}{params[1]:+.12g}*T\n"
    tdb += f"   {params[2]:+.12g}*T*LN(T){params[3]:+.12g}*T**2{params[4]:+.12g}*T**(-1)\n"
    tdb += f"   {params[5]:+.12g}*T**3; {Tlist[-1]:g} N !\n"
    tdb += "\n"
    # create TDB entry for end member x = 1
    res = scipy.optimize.curve_fit(tlogpoly, Tall, G1, np.ones(6))
    params = res[0] * R / kb
    tdb += f"{end[1]} {Tlist[0]:g} {params[0]:+.12g}{params[1]:+.12g}*T\n"
    tdb += f"   {params[2]:+.12g}*T*LN(T){params[3]:+.12g}*T**2{params[4]:+.12g}*T**(-1)\n"
    tdb += f"   {params[5]:+.12g}*T**3; {Tlist[-1]:g} N !\n"
    tdb += "\n"

    # fit redlich-kister parameters
    rkparams = []
    for iT, T in enumerate(Tall):
        ydata = []
        for ix, x in enumerate(xall):
            G_m0 = (1 - x) * G0[iT] + x * G1[iT]
            if (x == 0 or x == 1.):
                G_mix = 0
            else:
                G_mix = kb * T * (x * np.log(x) + (1 - x) * np.log(1 - x))
            ydata.append(Gall[iT, ix] - G_m0 - G_mix)
        xdata = np.append(xall, 1)
        ydata.append(0)
        # least square fitting
        res = scipy.optimize.curve_fit(
            redlich_kister, xdata, ydata, np.ones(4))
        rkparams.append(res[0])
    rkparams = np.array(rkparams)
    # fit log-poly function for T-dependence
    for i in range(4):
        res = scipy.optimize.curve_fit(
            tlogpoly, Tall, rkparams[:, i], np.ones(6))
        params = res[0] * R / kb
        tdb += f"{rk[i]} {Tlist[0]} {params[0]:+.12g}{params[1]:+.12g}*T\n"
        tdb += f"   {params[2]:+.12g}*T*LN(T){params[3]:+.12g}*T**2{params[4]:+.12g}*T**(-1)\n"
        tdb += f"   {params[5]:+.12g}*T**3; {Tlist[-1]} N !\n"
        tdb += "\n"
    return tdb


def process_solid(general, solid, write_file=True):
    '''
    postprocess solid calculations
    '''
    sol_dir = f"{general.proj_dir}/solid"
    if not os.path.isdir(sol_dir):
        exapd_logger.critical(f"{sol_dir} does not exist for post-processing.")

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
    # create a finer T-mesh for smooth free energy
    if general.units == "lj":
        ddT = 0.01
    else:
        ddT = 10
    Tall = np.arange(min(Tlist), max(Tlist) + 0.1 * ddT, ddT)
    tdb = ''  # entries for solid phases in the TDB file
    for ph in phases:
        # supports lammps format or other formats that can be converted
        # to lammps format by ASE.
        ph_file = os.path.abspath(ph)
        name, form = ph_file.split('/')[-1].split('.')
        phdir = f"{sol_dir}/{name}"
        if not os.path.isdir(phdir):
            raise Exception(f"{phdir} does not exist for post-processing job!")
        if form == "lammps":
            data_in = ph_file
        else:
            data_in = f"{phdir}/{name}.lammps"
        barostat = get_lammps_barostat(data_in)
        #  process tramp jobs
        sol_tramp = tramp(data_in, Tlist, f"{phdir}/tramp")
        res = sol_tramp.process()
        arrT, arrH = fix_enthalpy(res[:, 0], res[:, 1], phase="solid")
        # process einstein jobs
        natom, ntyp, nab = read_lmp_data(data_in, read_nab=True)
        pre_job_dir = f"{phdir}/tramp/T{Tlist[0]:g}"
        pre_var_names = "fxlo fxhi fylo fyhi fzlo fzhi".split()
        pre_var_values = "Xlo Xhi Ylo Yhi Zlo Zhi".split()
        if barostat == "tri":
            pre_var_names += "fxy fxz fyz".split()
            pre_var_values += "Xy Xz Yz".split()
        for i in range(ntyp):  # msd
            if nab[i] > 0:
                pre_var_names.append(f"msd{i+1}")
                pre_var_values.append(f"c_c{i+1}[4]")
        depend = (pre_job_dir, pre_var_names, pre_var_values)
        sol_ti = einstein(
            data_in, dlbd, Tlist[0], directory=f"{phdir}/einstein")
        # calculate G(T) using Gibbs-Helmholtz equation
        G0 = sol_ti.process(general, depend)
        Gall = Gibbs_Helmholtz(arrT, arrH, Tlist[0], G0, Tall)
        if write_file:
            np.savetxt(f"{general.proj_dir}/g.{name}.dat",
                       np.column_stack((Tall, Gall)), fmt="%.6f",
                       header=f"Gibbs free energy of the {name} phase\n   T      G")
        # fitting to log-poly function
        res = scipy.optimize.curve_fit(tlogpoly, Tall, Gall, np.ones(6))
        params = res[0]
        # create TDB entry
        tdb += create_tdb_header_nonsol(name, general.system, nab)
        tdb += f"{Tlist[0]:g} {params[0]:+.12g}{params[1]:+.12g}*T\n"
        tdb += f"   {params[2]:+.12g}*T*LN(T){params[3]:+.12g}*T**2{params[4]:+.12g}*T**(-1)\n"
        tdb += f"   {params[5]:+.12g}*T**3; {Tlist[-1]:g} N !\n"
        tdb += "\n"

    return tdb


def fix_enthalpy(arrT, arrH, phase):
    '''
    in T-ramping of a phase, crystallization or melting can occur,
    causing sudden change of H (kink).
    use linear extrapolation to get rid of the kink.
    arrT: temperature array;
    arrH: corresponding enthalpy data;
    phase: "liquid" or "solid", scan from high to low T for liquid to detect
    crystallization; and low to high T for solid to detect melting.
    '''
    idxsort = np.argsort(arrT)
    if phase == "solid":
        sortedT = np.asarray(arrT)[idxsort]
        sortedH = np.asarray(arrH)[idxsort]
    elif phase == "liquid":
        sortedT = np.asarray(arrT)[idxsort[::-1]]
        sortedH = np.asarray(arrH)[idxsort[::-1]]
    else:
        raise Exception("phase needs to be ""liquid"" or ""solid""!")

    dH = np.diff(sortedH) / np.diff(sortedT)
    for i in range(1, len(dH)):
        avg_prev = np.mean(dH[:i])
        if avg_prev == 0:
            continue
        # detect kink with a sudden change of slope by a factor > 2
        if abs(dH[i]) > 2.0 * abs(avg_prev):
            # linear interpolate to the last temperature
            sortedT[i + 1] = sortedT[-1]
            sortedH[i + 1] = sortedH[i] + dH[i - 1] * \
                (sortedT[i + 1] - sortedT[i])
            sortedT = sortedT[:i + 2]
            sortedH = sortedH[:i + 2]
            break
    if phase == "liquid":
        return sortedT[::-1], sortedH[::-1]
    else:
        return sortedT, sortedH


def Gibbs_Helmholtz(arrT, arrH, T0, G0, Tall):
    '''
    Compute Gibbs free energy from temperature-enthalpy data
    using Gibbs-Helmholtz equation.

    Parameters:
    arrT, arrH: input T and H values
    T0: Reference temperature.
    G0: Gibbs free energy at T0.
    Tall: strictly sorted temperature array to calculate Gibbs free energy

    Returns:
    Gall: G for Tall
    '''
    if not np.all(np.diff(Tall) > 0):
        raise Exception(
            "Tall should be strictly sorted in ascending order for G-H integration.")

    i0 = np.searchsorted(Tall, T0)
    if Tall[i0] == T0:
        Tint = Tall
    else:
        Tint = np.insert(Tall, i0, T0)

    # Interpolate enthalpy values using cubic spline
    # H = scipy.interpolate.interp1d(x, y, kind='cubic', fill_value='extrapolate')(T)
    interpolator = scipy.interpolate.PchipInterpolator(
        arrT, arrH, extrapolate=True)
    Hint = interpolator(Tint)
    # Initialize ΔG/T array
    detG_over_T = np.zeros(len(Tint))
    G0_over_T0 = G0 / T0

    # Compute Gibbs free energy difference over temperature
    for i in range(i0):
        detG_over_T[i] = scipy.integrate.simpson(
            Hint[i:i0 + 1] / Tint[i:i0 + 1]**2, Tint[i:i0 + 1])
    for i in range(i0, len(Tint)):
        detG_over_T[i] = -scipy.integrate.simpson(
            Hint[i0:i + 1] / Tint[i0:i + 1]**2, Tint[i0:i + 1])

    # Compute absolute Gibbs free energy
    Gall = (detG_over_T + G0_over_T0) * Tint
    if len(Gall) == len(Tall):
        return Gall
    else:
        return np.delete(Gall, i0)


def redlich_kister(x, l0, l1, l2, l3):
    '''
    Redlich-Kister polynomial for regular mixing
    '''
    return (l0 + l1 * (1 - 2 * x) + l2 * (1 - 2 * x)**2 + l3 * (1 - 2 * x)**3) * x * (1 - x)


def tlogpoly(T, l0, l1, l2, l3, l4, l5):
    '''
    logrithmic-polynomial function for T dependence
    '''
    return l0 + l1 * T + l2 * T * np.log(T) + l3 * T**2 + l4 / T + l5 * T**3
