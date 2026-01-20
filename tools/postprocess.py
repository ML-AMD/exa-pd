from jobs.lammpsJob import *
from jobs.alchem import *
from jobs.einstein import *
from jobs.tramp import *
import matplotlib.pyplot as plt
from tools.logging_config import exapd_logger
from tools.utils import merge_arrays
import os


def process_liquid(general, liquid, write_file=True):
    """
    Post-process liquid free-energy calculations and optionally generate a TDB entry.

    This routine processes a set of composition points between an initial and final
    composition using:
    - alchemical thermodynamic integration (via :class:`jobs.alchem.alchem`) to obtain
      the Gibbs free-energy offset at a reference temperature, and
    - temperature ramping (via :class:`jobs.tramp.tramp`) to obtain enthalpy vs. T,
      which is integrated using the Gibbs–Helmholtz relation.

    It writes a grid of G(T, x) to ``g.liq.dat`` and, when applicable, constructs a
    CALPHAD TDB entry for a (sub)binary liquid.

    Parameters
    ----------
    general : lammpsPara
        General simulation parameters (units, elements, project directory, etc.).
    liquid : dict
        Liquid calculation settings. Expected keys include (not all are mandatory):

        - ``data_in`` : str
            LAMMPS data file for the structure.
        - ``initial_comp`` : array-like
            Initial composition vector.
        - ``final_comp`` : array-like
            Final composition vector.
        - ``ncomp`` : int, optional
            Number of composition steps between initial and final composition.
        - ``ref_pair_style`` / ``ref_pair_coeff`` : optional
            Reference potential used for TI at compositions other than the initial one.
        - ``dlbd`` : float, optional
            $lambda$ spacing for TI.
        - ``Tlist`` or ``Tmin``/``Tmax``/``dT`` : optional
            Temperatures for ramping and defining the TI reference temperature.

    write_file : bool, optional
        If True, write ``g.liq.dat`` into ``general.proj_dir``.

    Returns
    -------
    str
        TDB text for the liquid phase if a (sub)binary system is detected; otherwise
        an empty string.

    Notes
    -----
    - The method assumes the project directory contains a ``liquid`` folder with
      subfolders ``comp0``, ``comp1``, ... that contain completed TI and ramp runs.
    - For the initial composition point, UFM is used as the default reference.
    """
    liq_dir = f"{general.proj_dir}/liquid"
    if not os.path.isdir(liq_dir):
        exapd_logger.critical(
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
    R = 8.3144598  # gas constant
    Tall = np.arange(min(Tlist), max(Tlist) + 0.1 * ddT, ddT)
    tdb = ''  # entry of the liquid phase in the TDB file
    natom, ntyp = read_lmp_data(data_in)
    n0 = natom * comp0
    n1 = natom * comp1
    dn = (n1 - n0) / ncomp
    xall = np.zeros(ncomp + 1)
    comp_idx = 0
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
            rexapd_logger.critical(
                f"{compdir} does not exist for post-processing.")
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
                G0 = det_G  # det_G is absolute for using UFM as ref
            else:
                # det_G is relative G to G(x=0) otherwise
                G0 = det_G + Gall[-1, 0]
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

    phase, end, rk = create_tdb_binsol_phase("liq", *binary)
    tdb += phase  # phase definition in TDB
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
    """
    Post-process solid-phase free-energy calculations and generate TDB entries.

    For each solid phase structure file provided, this routine:
    - post-processes temperature-ramping results to obtain H(T),
    - runs the Frenkel–Ladd (Einstein crystal) workflow to obtain an absolute
      free-energy reference at a chosen temperature, and
    - integrates G(T) using the Gibbs–Helmholtz relation and fits it to a
      log-polynomial form to generate a TDB entry.

    Parameters
    ----------
    general : lammpsPara
        General simulation parameters (units, elements, project directory, etc.).
    solid : dict
        Solid calculation settings. Expected keys include:

        - ``phases`` : list of str
            Paths to structure files defining phases.
        - ``dlbd`` : float, optional
            λ spacing for the Einstein TI.
        - ``Tlist`` or ``Tmin``/``Tmax``/``dT`` : optional
            Temperatures used for ramping and for selecting the TI reference temperature.

    write_file : bool, optional
        If True, writes ``g.<phase>.dat`` in ``general.proj_dir`` for each phase.

    Returns
    -------
    str
        Concatenated TDB entries for all processed solid phases.
    """
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
        kb = 1
        ddT = 0.01
    else:
        kb = 8.617333262e-5
        ddT = 10
    R = 8.3144598  # gas constant
    Tall = np.arange(min(Tlist), max(Tlist) + 0.1 * ddT, ddT)
    tdb = ''  # entries for solid phases in the TDB file
    for ph in phases:
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
        sol_tramp = tramp(data_in, Tlist, f"{phdir}/tramp")
        res = sol_tramp.process()
        arrT, arrH = fix_enthalpy(res[:, 0], res[:, 1], phase="solid")
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
        sol_ti = einstein(
            data_in, dlbd, Tlist[0], directory=f"{phdir}/einstein")
        G0 = sol_ti.process(general, depend)
        Gall = Gibbs_Helmholtz(arrT, arrH, Tlist[0], G0, Tall)
        if write_file:
            np.savetxt(f"{general.proj_dir}/g.{name}.dat",
                       np.column_stack((Tall, Gall)), fmt="%.6f",
                       header=f"Gibbs free energy of the {name} phase\n   T      G")
        res = scipy.optimize.curve_fit(tlogpoly, Tall, Gall, np.ones(6))
        params = res[0] * R / kb
        tdb += create_tdb_nonsol_phase(name, general.system, nab)
        tdb += f"{Tlist[0]:g} {params[0]:+.12g}{params[1]:+.12g}*T\n"
        tdb += f"   {params[2]:+.12g}*T*LN(T){params[3]:+.12g}*T**2{params[4]:+.12g}*T**(-1)\n"
        tdb += f"   {params[5]:+.12g}*T**3; {Tlist[-1]:g} N !\n\n"

    return tdb


def fix_enthalpy(arrT, arrH, phase):
    """
    Detect and remove enthalpy kinks caused by phase transformations during ramping.

    During temperature ramping, crystallization (liquid) or melting (solid) can
    produce a sudden slope change in enthalpy vs temperature. This function detects
    a kink by a large slope change and then applies a linear extrapolation to
    smooth the data beyond the kink.

    Parameters
    ----------
    arrT : array-like
        Temperature samples.
    arrH : array-like
        Enthalpy samples corresponding to `arrT`.
    phase : {"liquid", "solid"}
        Phase type that determines scan direction:
        - ``"liquid"`` scans from high to low temperature (crystallization detection),
        - ``"solid"`` scans from low to high temperature (melting detection).

    Returns
    -------
    tuple of numpy.ndarray
        (T_smooth, H_smooth) arrays after kink removal.

    Raises
    ------
    Exception
        If `phase` is not one of ``"liquid"`` or ``"solid"``.
    """
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
        if abs(dH[i]) > 2.0 * abs(avg_prev):
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
    """
    Compute Gibbs free energy G(T) from enthalpy data using Gibbs–Helmholtz integration.

    Parameters
    ----------
    arrT : array-like
        Temperature samples.
    arrH : array-like
        Enthalpy samples at `arrT`.
    T0 : float
        Reference temperature where G(T0) is known.
    G0 : float
        Gibbs free energy at the reference temperature `T0`.
    Tall : array-like
        Strictly increasing temperature grid for evaluating G(T).

    Returns
    -------
    numpy.ndarray
        Gibbs free energy evaluated at temperatures in `Tall`.

    Raises
    ------
    Exception
        If `Tall` is not strictly increasing.

    Notes
    -----
    The implementation uses a monotone PCHIP interpolator for H(T) and Simpson
    integration of H(T)/T^2.
    """
    if not np.all(np.diff(Tall) > 0):
        raise Exception(
            "Tall should be strictly sorted in ascending order for G-H integration.")

    i0 = np.searchsorted(Tall, T0)
    if Tall[i0] == T0:
        Tint = Tall
    else:
        Tint = np.insert(Tall, i0, T0)

    interpolator = scipy.interpolate.PchipInterpolator(
        arrT, arrH, extrapolate=True)
    Hint = interpolator(Tint)
    detG_over_T = np.zeros(len(Tint))
    G0_over_T0 = G0 / T0

    for i in range(i0):
        detG_over_T[i] = scipy.integrate.simpson(
            Hint[i:i0 + 1] / Tint[i:i0 + 1]**2, Tint[i:i0 + 1])
    for i in range(i0, len(Tint)):
        detG_over_T[i] = -scipy.integrate.simpson(
            Hint[i0:i + 1] / Tint[i0:i + 1]**2, Tint[i0:i + 1])

    Gall = (detG_over_T + G0_over_T0) * Tint
    if len(Gall) == len(Tall):
        return Gall
    else:
        return np.delete(Gall, i0)


def redlich_kister(x, l0, l1, l2, l3):
    """
    Redlich–Kister polynomial for excess mixing free energy.

    Parameters
    ----------
    x : float or array-like
        Composition variable (typically mole fraction of one component).
    l0, l1, l2, l3 : float
        Redlich–Kister coefficients.

    Returns
    -------
    float or numpy.ndarray
        Excess mixing contribution at composition `x`.
    """
    return (l0 + l1 * (1 - 2 * x) + l2 * (1 - 2 * x)
            ** 2 + l3 * (1 - 2 * x)**3) * x * (1 - x)


def tlogpoly(T, l0, l1, l2, l3, l4, l5):
    """
    Log-polynomial function for temperature dependence.

    Parameters
    ----------
    T : float or array-like
        Temperature.
    l0, l1, l2, l3, l4, l5 : float
        Coefficients of the log-polynomial form.

    Returns
    -------
    float or numpy.ndarray
        Function value at temperature(s) `T`.
    """
    return l0 + l1 * T + l2 * T * np.log(T) + l3 * T**2 + l4 / T + l5 * T**3
