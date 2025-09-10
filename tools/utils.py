import numpy as np
import os


def read_lmp_data(data_in, read_nab=False, read_pos=False):
    '''
    data_in: lammps input data file to be read.
    if read_nab: output the number of atoms in each type [na, nb, ...];
    elif read_pos: output pos containing cartesian coordinates of atoms;
    else: only output total number of atoms and total number of types.
    '''
    if not os.path.exists(data_in):
        raise Exception(f"{infile} doesn't exist!")
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
                    type_id = int(fields[1])  # For 'atomic' style
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
    '''
    determine what barostat to use. return tri if xy xz yz in file
    data_in: input lammps file
    eps: tolerance for lengths
    '''
    if not os.path.exists(data_in):
        raise Exception(f"{infile} doesn't exist!")
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
            if a is not None and b is not None and c is not None:
                break
    if a is None or b is None or c is None:
        raise Exception(f"Incomplete lammps input file {data_in}!")
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
    '''
    create a supercell in lammps format from a crystal structure file.
    system: list of elements in the system
    infile: crystal sturcute file
    directory: location for the output file
    outfile: path to output lammps file
    ntarget: approximate number of atoms in the lammps file
    eps: tolerance for lengths & angles
    '''
    from ase.io import read, write
    from ase.io.lammpsdata import write_lammps_data
    from ase.build.supercells import make_supercell
    if not os.path.exists(infile):
        raise Exception(f"{infile} doesn't exist!")

    # read crystal structure using ASE, convert hex or trigonal to ortho
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
    elif abs(a - b) < eps and abs(alpha - 90) < eps \
            and abs(beta - 90) < eps and abs(gamma - 120) < eps:  # hexagonal cell
        structure = make_supercell(structure, [[1, -1, 0], [1, 1, 0], [0, 0, 1]])
        cell = structure.get_cell()
        a, b, c = cell.lengths()
        barostat = "couple xy"
    elif abs(a - b) < eps and abs(alpha - 90) < eps \
            and abs(beta - 90) < eps and abs(gamma - 60) < eps:  # hexagonal cell
        structure = make_supercell(structure, [[1, 1, 0], [-1, 1, 0], [0, 0, 1]])
        cell = structure.get_cell()
        a, b, c = cell.lengths()
        barostat = "couple xy"
    ### trigonal cell to be added ###

    # generate supercell
    rho = len(structure) / structure.get_volume()
    boxsize = (ntarget / rho) ** (1 / 3)
    na = round(boxsize / a)
    nb = round(boxsize / b)
    nc = round(boxsize / c)
    supercell = structure * (na, nb, nc)

    # write to lammps file
    name = infile.split('/')[-1].split('.')[0]
    # write_lammps_data(outfile, supercell, atom_style="atomic")
    types = supercell.get_chemical_symbols()
    frac_coords = supercell.get_scaled_positions()
    cell = supercell.get_cell()
    a, b, c = cell.lengths()
    alpha, beta, gamma = cell.angles()
    try:
        f = open(outfile, "wt")
    except BaseException:
        print(f"cannot open {outfile} to write!")
        raise
    f.write(f"generated from {infile}\n")
    f.write("\n")
    f.write(f"{len(types)} atoms\n")
    f.write(f"{len(system)} atom types\n")
    f.write("\n")

    if abs(alpha - 90) < eps and abs(beta - 90) < eps and abs(gamma - 90) < eps:
        lx, ly, lz = a, b, c
        xy = xz = yz = 0
        f.write(f"0.0      {lx:.6f} xlo xhi\n")
        f.write(f"0.0      {ly:.6f} ylo yhi\n")
        f.write(f"0.0      {lz:.6f} zlo zhi\n")
    else:  # triclinic cell
        lx, ly, lz, xy, xz, yz = create_triclinic_box(a, b, c, alpha, beta, gamma)
        f.write(f"0.0      {lx:.6f} xlo xhi\n")
        f.write(f"0.0      {ly:.6f} ylo yhi\n")
        f.write(f"0.0      {lz:.6f} zlo zhi\n")
        f.write(f"{xy:.6f} {xz:.6f} {yz:.6f} xy xz yz\n")
    f.write("\n")
    f.write("Atoms # atomic\n")
    f.write("\n")
    for i in range(len(types)):
        if types[i] in system:
            fx, fy, fz = frac_coords[i]
            x = fx * lx + fy * xy + fz * xz
            y = fy * ly + fz * yz
            z = fz * lz
            f.write(f"{i+1:6d} {system.index(types[i])+1:4d} {x:16.6f} {y:16.6f} {z:16.6f}\n")
        else:
            raise Exception(f"element {typ} is not in system in {infile}!")
    f.close()
    return barostat


def create_triclinic_box(a, b, c, alpha, beta, gamma, radians=False):
    '''
    convert conventional lattice parameters to box parameters for lammps.
    alpha, beta, gamma are in units of degrees by default
    set radians=True if in radians already.
    '''
    if not radians:
        alpha_rad = np.radians(alpha)
        beta_rad = np.radians(beta)
        gamma_rad = np.radians(gamma)

    # Calculate cosines and sine
    cos_alpha = np.cos(alpha_rad)
    cos_beta = np.cos(beta_rad)
    cos_gamma = np.cos(gamma_rad)
    sin_gamma = np.sin(gamma_rad)

    # Compute components of lattice paremeters
    lx = a
    ly = b * sin_gamma
    lz = c * np.sqrt(1 - cos_alpha**2 - cos_beta**2 - cos_gamma**2 +
                     2 * cos_alpha * cos_beta * cos_gamma) / sin_gamma
    xy = b * cos_gamma
    xz = c * cos_beta
    yz = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma

    return lx, ly, lz, xy, xz, yz


def create_tdb_header_nonsol(name, system, nab):
    '''
    create the TDB header for a non-solution phase.
    name, str: name of the phase
    system, list: ['A', 'B',...], elements in a system
    nab, list: [na, nb,...], number of atoms for each element,
    '''
    ntyp = len(system)
    natom = sum(nab)
    tdb = ''
    comp = {}  # composition of the phase
    for i in range(ntyp):
        if nab[i] > 0:
            comp[system[i]] = nab[i] / natom
    tdb += f" PHASE {name}  %  {len(comp)}  "
    for el in comp.keys():
        tdb += f"{comp[el]:.6f} "
    tdb += "!\n"
    tdb += f"    CONSTITUENT {name}  :"
    for el in comp.keys():
        tdb += f"{el} : "
    tdb += "!\n\n"
    tdb += f"   PARAMETER G({name},"
    for el in comp.keys():
        tdb += f"{el}:"
    tdb = tdb[:-1] + ";0) "
    return tdb


def create_tdb_header_binsol(name, el1, el2):
    '''
    create the TDB header for a binary solution phase with one sublattice.
    name, str: name of the phase
    el1, str:: name of the first element in the solution
    el2, str: number of the second element in the solution,
    '''
    # definition of the phase
    phase = f" PHASE {name}  %  1 1.0 !\n"
    phase += f"    CONSTITUENT {name}  :{el1},{el2} : !\n\n"

    # params for endmembers
    end = ['', '']
    end[0] = (f"   PARAMETER G({name},{el1};0)")
    end[1] = f"   PARAMETER G({name},{el2};0)"

    # params for R-K coefficients L0, L1, L2, L3
    rk = [''] * 4
    for i in range(4):
        rk[i] = f"   PARAMETER L({name},{el1},{el2};{i})"

    return phase, end, rk


def merge_arrays(arr1, arr2, tolerance=0.001):
    # Combine the arrays
    combined = np.concatenate((arr1, arr2))
    
    # Sort the combined array
    combined = np.sort(combined)
    
    # Remove duplicates within tolerance
    result = [combined[0]]
    for i in range(1, len(combined)):
        if abs(combined[i] - result[-1]) > tolerance:
            result.append(combined[i])
            
    return np.array(result)
