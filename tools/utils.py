import numpy as np
from tools.logging_config import exapd_logger
import os


def read_lmp_data(data_in, read_nab=False, read_pos=False):
    """
    Read a LAMMPS data file and extract basic structural information.

    Parameters
    ----------
    data_in : str
        Path to the LAMMPS data file.
    read_nab : bool, optional
        If True, return the number of atoms of each type.
    read_pos : bool, optional
        If True, return atomic Cartesian positions.

    Returns
    -------
    tuple
        Depending on flags:

        - (natom, ntyp)
        - (natom, ntyp, nab)
        - (natom, ntyp, pos)
        - (natom, ntyp, nab, pos)

    Notes
    -----
    The function assumes ``atom_style atomic`` formatting.
    """
    '''
    data_in: lammps input data file to be read.
    if read_nab: output the number of atoms in each type [na, nb, ...];
    elif read_pos: output pos containing cartesian coordinates of atoms;
    else: only output total number of atoms and total number of types.
    '''
    if not os.path.exists(data_in):
        exapd_logger.critical(f"{data_in} does not exist.")
    natom = ntyp = None
    reading_atoms = False
    with open(data_in) as infile:
        for line in infile:
            line = line.strip()
            if "atoms" in line:
                natom = int(line.split()[0])
            if "atom types" in line:
                ntyp = int(line.split()[0])
            if natom is not None and ntyp is not None and not read_nab \
                    and not read_pos:
                return natom, ntyp
            if line.startswith('Atoms'):
                reading_atoms = True
                if read_nab:
                    nab = np.zeros(ntyp, dtype=int)
                if read_pos:
                    pos = np.zeros((natom, 3))
                count = 0
                continue
            if reading_atoms and line and count < natom:
                fields = line.split()
                if read_nab:
                    type_id = int(fields[1])
                    nab[type_id - 1] += 1
                if read_pos:
                    pos[count, :] = [float(value) for value in fields[2:5]]
                count += 1
    if read_nab and read_pos:
        return natom, ntyp, nab, pos
    elif read_nab:
        return natom, ntyp, nab
    else:
        return natom, ntyp, pos


def get_lammps_barostat(data_in, eps=1e-3):
    """
    Determine the appropriate LAMMPS barostat style from a data file.

    Parameters
    ----------
    data_in : str
        Path to the LAMMPS data file.
    eps : float, optional
        Tolerance for comparing lattice parameters.

    Returns
    -------
    str
        Barostat type: ``"iso"``, ``"aniso"``, ``"tri"``, or ``"couple xy/xz/yz"``.
    """
    '''
    determine what barostat to use. return tri if xy xz yz in file
    data_in: input lammps file
    eps: tolerance for lattice parameters
    '''
    if not os.path.exists(data_in):
        exapd_logger.critical(f"{data_in} does not exist.")
    a = b = c = None
    with open(data_in) as infile:
        for line in infile:
            if "xy" in line and "xz" in line and "yz" in line:
                return "tri"
            if "xlo" in line and "xhi" in line:
                values = line.split()
                a = float(values[1]) - float(values[0])
            if "ylo" in line and "yhi" in line:
                values = line.split()
                b = float(values[1]) - float(values[0])
            if "zlo" in line and "zhi" in line:
                values = line.split()
                c = float(values[1]) - float(values[0])
            if "Atoms" in line:
                break
    if a is None or b is None or c is None:
        exapd_logger.critical(f"Incomplete lammps input file {data_in}.")
    diffab = abs(a - b)
    diffac = abs(a - c)
    diffbc = abs(b - c)
    if diffab < eps and diffac < eps and diffbc < eps:
        return "iso"
    elif diffab < eps:
        return "couple xy"
    elif diffac < eps:
        return "couple xz"
    elif diffbc < eps:
        return "couple yz"
    else:
        return "aniso"


def create_lammps_supercell(system, infile, outfile, ntarget=500, eps=1.e-3):
    """
    Create a LAMMPS data file for a supercell built from a crystal structure.

    Parameters
    ----------
    system : list of str
        Chemical symbols defining the system ordering.
    infile : str
        Input crystal structure file readable by ASE.
    outfile : str
        Output LAMMPS data file path.
    ntarget : int, optional
        Target number of atoms in the supercell.
    eps : float, optional
        Tolerance for lattice parameter comparisons.

    Returns
    -------
    str
        Suggested barostat type for the generated supercell.
    """
    '''
    create a supercell in lammps format from a crystal structure file.
    '''
    from ase.io import read
    from ase.build.supercells import make_supercell
    if not os.path.exists(infile):
        exapd_logger.critical(f"{infile} does not exist.")

    structure = read(infile)
    cell = structure.get_cell()
    a, b, c = cell.lengths()
    alpha, beta, gamma = cell.angles()
    barostat = "tri"
    if abs(alpha - 90) < eps and abs(beta - 90) < eps and abs(gamma - 90) < eps:
        if abs(a - b) < eps and abs(a - c) < eps and abs(b - c) < eps:
            barostat = "iso"
        elif abs(a - b) < eps:
            barostat = "couple xy"
        elif abs(a - c) < eps:
            barostat = "couple xz"
        elif abs(b - c) < eps:
            barostat = "couple yz"
        else:
            barostat = "aniso"

    rho = len(structure) / structure.get_volume()
    boxsize = (ntarget / rho) ** (1 / 3)
    na = round(boxsize / a)
    nb = round(boxsize / b)
    nc = round(boxsize / c)
    supercell = structure * (na, nb, nc)

    cell = supercell.get_cell()
    a, b, c = cell.lengths()
    alpha, beta, gamma = cell.angles()
    if abs(alpha - 90) < eps and abs(beta - 90) < eps and abs(gamma - 90) < eps:
        lx, ly, lz = a, b, c
        xy = xz = yz = 0
    else:
        lx, ly, lz, xy, xz, yz, rotmat = create_triclinic_box(
            a, b, c, alpha, beta, gamma)
        if rotmat is not None:
            supercell = make_supercell(supercell, rotmat)

    name = infile.split('/')[-1].split('.')[0]
    types = supercell.get_chemical_symbols()
    frac_coords = supercell.get_scaled_positions()

    try:
        f = open(outfile, "wt")
    except Exception:
        exapd_logger.critical(f"Cannot open {outfile} to write.")

    f.write(f"generated from {infile}\n\n")
    f.write(f"{len(types)} atoms\n")
    f.write(f"{len(system)} atom types\n\n")
    f.write(f"0.0 {lx:.6f} xlo xhi\n")
    f.write(f"0.0 {ly:.6f} ylo yhi\n")
    f.write(f"0.0 {lz:.6f} zlo zhi\n")
    if xy != 0 or xz != 0 or yz != 0:
        f.write(f"{xy:.6f} {xz:.6f} {yz:.6f} xy xz yz\n")
    f.write("\nAtoms # atomic\n\n")

    for i, el in enumerate(types):
        if el not in system:
            exapd_logger.critical(
                f"Element {el} is not in system in {infile}!")
        fx, fy, fz = frac_coords[i]
        x = fx * lx + fy * xy + fz * xz
        y = fy * ly + fz * yz
        z = fz * lz
        f.write(
            f"{i + 1:6d} {system.index(el) + 1:4d} "
            f"{x:16.6f} {y:16.6f} {z:16.6f}\n"
        )
    f.close()
    return barostat


def create_triclinic_box(a, b, c, alpha, beta, gamma, radians=False):
    """
    Convert lattice parameters to LAMMPS triclinic box parameters.

    Parameters
    ----------
    a, b, c : float
        Lattice lengths.
    alpha, beta, gamma : float
        Lattice angles.
    radians : bool, optional
        If True, angles are already in radians.

    Returns
    -------
    tuple
        (lx, ly, lz, xy, xz, yz, rotmat)
    """
    '''
    convert conventional lattice parameters to box parameters for lammps.
    '''
    if not radians:
        alpha = np.radians(alpha)
        beta = np.radians(beta)
        gamma = np.radians(gamma)

    cos_alpha = np.cos(alpha)
    cos_beta = np.cos(beta)
    cos_gamma = np.cos(gamma)
    sin_gamma = np.sin(gamma)

    lx = a
    ly = b * sin_gamma
    lz = c * np.sqrt(
        1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2
        + 2 * cos_alpha * cos_beta * cos_gamma
    ) / sin_gamma
    xy = b * cos_gamma
    xz = c * cos_beta
    yz = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma

    if abs(xy) <= lx / 2 and abs(xz) <= lx / 2 and abs(yz) <= ly / 2:
        return lx, ly, lz, xy, xz, yz, None
    else:
        return fix_large_skew(lx, ly, lz, xy, xz, yz)


def fix_large_skew(lx, ly, lz, xy, xz, yz):
    """
    Reduce large skew components of a triclinic box via basis rotation.

    Returns
    -------
    tuple
        (lx, ly, lz, xy, xz, yz, rotation_matrix)
    """
    rotmat = np.eye(3)
    while abs(yz) > ly / 2:
        sgn = np.sign(yz)
        rotmat[2] -= sgn * rotmat[1]
        yz -= sgn * ly
        xz -= sgn * xy
    while abs(xz) > lx / 2:
        sgn = np.sign(xz)
        rotmat[2] -= sgn * rotmat[0]
        xz -= sgn * lx
    while abs(xy) > lx / 2:
        sgn = np.sign(xy)
        rotmat[1] -= sgn * rotmat[0]
        xy -= sgn * lx
    return lx, ly, lz, xy, xz, yz, rotmat


def create_tdb_header(system, mass):
    """
    Create the global header of a CALPHAD TDB file.

    Parameters
    ----------
    system : list of str
        Chemical symbols.
    mass : list of float
        Atomic masses.

    Returns
    -------
    str
        Uppercase TDB header text.
    """
    '''
    create overall header of the TDB file
    '''
    tdb = ''
    tdb += f"$ DATA FILE {' '.join(system)}\n"
    tdb += "ELEMENT /-   ELECTRON_GAS              0.0000E+00  0.0000E+00  0.0000E+00!\n"
    tdb += "ELEMENT VA   VACUUM                    0.0000E+00  0.0000E+00  0.0000E+00!\n"
    for el, m in zip(system, mass):
        tdb += f"ELEMENT {el} NA                          {m:.4E}  0.0000E+00  0.0000E+00!\n"
    tdb += "\n TYPE_DEFINITION % SEQ *!\n"
    tdb += f" DEFINE_SYSTEM_DEFAULT ELEMENT {len(system)} !\n"
    tdb += " DEFAULT_COMMAND DEF_SYS_ELEMENT VA /- !\n\n"
    return tdb.upper()


def create_tdb_nonsol_phase(name, system, nab):
    """
    Create a TDB phase header for a non-solution phase.

    Parameters
    ----------
    name : str
        Phase name.
    system : list of str
        Chemical symbols.
    nab : list of int
        Number of atoms of each species.

    Returns
    -------
    str
        TDB phase header.
    """
    '''
    create the TDB header for a non-solution phase.
    '''
    natom = sum(nab)
    comp = {system[i]: nab[i] / natom
            for i in range(len(system)) if nab[i] > 0}

    tdb = f" PHASE {name}  %  {len(comp)}  "
    for x in comp.values():
        tdb += f"{x:.6f} "
    tdb += "!\n"
    tdb += f"    CONSTITUENT {name}  :"
    for el in comp:
        tdb += f"{el} : "
    tdb += "!\n\n   PARAMETER G({name},"
    for el in comp:
        tdb += f"{el}:"
    tdb = tdb[:-1] + ";0) "
    return tdb.upper()


def create_tdb_binsol_phase(name, el1, el2):
    """
    Create a TDB header for a binary solution phase.

    Parameters
    ----------
    name : str
        Phase name.
    el1, el2 : str
        Element symbols.

    Returns
    -------
    tuple
        (phase_definition, endmember_params, rk_params)
    """
    '''
    create the TDB header for a binary solution phase with one sublattice.
    '''
    phase = f" PHASE {name}  %  1 1.0 !\n"
    phase += f"    CONSTITUENT {name}  :{el1},{el2} : !\n\n"
    phase = phase.upper()

    end = [
        f"   PARAMETER G({name},{el1};0)".upper(),
        f"   PARAMETER G({name},{el2};0)".upper()
    ]

    rk = [f"   PARAMETER L({name},{el1},{el2};{i})".upper()
          for i in range(4)]

    return phase, end, rk


def merge_arrays(arr1, arr2, tolerance=0.001):
    """
    Merge two numeric arrays while removing near-duplicates.

    Parameters
    ----------
    arr1, arr2 : array-like
        Input arrays.
    tolerance : float, optional
        Minimum separation to treat values as distinct.

    Returns
    -------
    numpy.ndarray
        Sorted merged array.
    """
    combined = np.sort(np.concatenate((arr1, arr2)))
    result = [combined[0]]
    for v in combined[1:]:
        if abs(v - result[-1]) > tolerance:
            result.append(v)
    return np.array(result)


def F_idealgas(temp, rho, natom, mass, comp, constants):
    """
    Compute the ideal-gas Helmholtz free energy per particle.

    Parameters
    ----------
    temp : float
        Temperature.
    rho : float
        Number density.
    natom : int
        Number of atoms.
    mass : list of float
        Atomic masses.
    comp : list of float
        Composition fractions.
    constants : tuple
        Physical constants (kb, hbar, au).

    Returns
    -------
    float
        Ideal-gas free energy.
    """
    '''
    free energy of ideal gas
    '''
    kb, hbar, au = constants
    beta = 1 / (kb * temp)
    debroglie = [
        np.sqrt(2 * np.pi * beta * hbar**2 / (m * au))
        for m in mass
    ]

    FE = 0
    for x, lam in zip(comp, debroglie):
        if x > 0:
            FE += x * (3 * np.log(lam) + np.log(rho) - 1 + np.log(x))
    return FE / beta
