import numpy as np
import os
from jobs.lammpsJob import *
from tools.utils import *
import sys


class tramp(lammpsJobGroup):
    """
    Temperature-ramping LAMMPS workflow for solid or liquid phases.

    This class sets up a series of independent LAMMPS simulations at
    different temperatures to equilibrate a structure and extract
    thermodynamic quantities such as enthalpy and volume.

    Parameters
    ----------
    data_in : str
        Initial structure file in LAMMPS data format.
    Tlist : list of float
        List of temperatures at which to equilibrate the system.
    directory : str
        Path to the group directory where job subfolders are created.
    nab : list of int, optional
        Number of atoms of each type [n1, n2, ...]. If provided, atomic
        types in `data_in` will be reassigned accordingly.
    barostat : str, optional
        Barostat type for NPT simulations. If set to `"none"`, the script
        still writes an `npt` fix line but without barostat coupling.
    """

    def __init__(self,
                 data_in,
                 Tlist,
                 directory,
                 nab=None,
                 barostat="iso"):
        super().__init__(directory)
        self._datain = data_in
        self._Tlist = Tlist
        self._barostat = barostat
        if nab is not None:
            natom, ntyp = read_lmp_data(self._datain)
            self._resetTypes = True
        else:
            natom, ntyp, nab = read_lmp_data(self._datain, read_nab=True)
            self._resetTypes = False
        self._natom = natom
        self._ntyp = ntyp
        self._nab = nab

    def setup(self, general, boxdims=False, msd=False):
        """
        Set up temperature-ramping LAMMPS jobs.

        Parameters
        ----------
        general : lammpsPara
            General LAMMPS parameters (units, pair potential, neighbor settings,
            masses, timestep, thermo frequency, pressure, Tdamp/Pdamp, run length).
        boxdims : bool, optional
            If True, output detailed box dimensions (xlo/xhi, etc.) instead of
            only volume. Default is False.
        msd : bool, optional
            If True, compute mean-squared displacement (MSD) for each atomic
            species. Default is False.
        """
        natom = self._natom
        for T in self._Tlist:
            Tdir = f"{self._dir}/T{T:g}"
            scriptFile = f"{Tdir}/lmp.in"
            job = lammpsJob(directory=Tdir,
                            scriptFile=scriptFile)
            if not os.path.exists(scriptFile):
                self.write_script(job._script, general, T, boxdims, msd)
            self._jobList.append(job)

    def write_script(self, scriptFile, general, T, boxdims, msd):
        """
        Write a LAMMPS input script for equilibration at a single temperature.

        Parameters
        ----------
        scriptFile : str
            Output path for the LAMMPS input script.
        general : lammpsPara
            General LAMMPS parameters and pair potential definition.
        T : float
            Temperature at which the system is equilibrated.
        boxdims : bool
            Whether to output detailed box dimensions.
        msd : bool
            Whether to compute mean-squared displacement (MSD).
        """
        f = open(scriptFile, 'wt')
        f.write(f"#  Equilibrate the structure for T = {T}\n\n")
        f.write(f"units           {general.units}\n")
        f.write("boundary        p p p\n")
        f.write("atom_style      atomic\n")
        f.write("atom_modify     map array\n\n")
        f.write(f"read_data       {self._datain}\n\n")

        if self._resetTypes:
            f.write(reset_types(self._nab, self._natom))

        f.write(general.pair._cmd)

        if general.neighbor is not None:
            f.write(f"neighbor        {general.neighbor}\n")
        f.write(f"neigh_modify    {general.neigh_modify}\n\n")

        if general.mass is not None:
            if isinstance(general.mass, list):
                for i in range(len(general.mass)):
                    f.write(f"mass            {i + 1} {general.mass[i]}\n")
            else:
                f.write(f"mass            * {general.mass}\n")
        f.write("\n")

        f.write(
            f"velocity        all create {T:g} "
            f"{np.random.randint(1000000)} rot yes dist gaussian\n"
        )

        if general.timestep is not None:
            f.write(f"timestep        {general.timestep}\n")

        f.write(f"thermo          {general.thermo}\n")

        if not boxdims:
            thermo_style = "custom step temp etotal press vol enthalpy"
        elif self._barostat == "tri":
            thermo_style = (
                "custom step temp etotal press "
                "xlo xhi ylo yhi zlo zhi xy xz yz enthalpy"
            )
        else:
            thermo_style = (
                "custom step temp etotal press "
                "xlo xhi ylo yhi zlo zhi enthalpy"
            )

        if self._barostat == "none":
            baro_style = ''
        elif "couple" not in self._barostat:
            baro_style = (
                f"{self._barostat} {general.pressure} "
                f"{general.pressure} {general.Pdamp}"
            )
        else:
            baro_style = (
                f"x {general.pressure} {general.pressure} {general.Pdamp} "
                f"y {general.pressure} {general.pressure} {general.Pdamp} "
                f"z {general.pressure} {general.pressure} {general.Pdamp} "
                + self._barostat
            )

        f.write(
            f"fix             1 all npt temp "
            f"{T:g} {T:g} {general.Tdamp} {baro_style}\n"
        )

        if msd:
            f.write("\n")
            f.write(f"run             {int(0.1 * general.run)}\n")
            for i in range(self._ntyp):
                if self._nab[i] > 0:
                    f.write(f"group           g{i + 1} type {i + 1}\n")
                    f.write(f"compute         c{i + 1} g{i + 1} msd com yes\n")
                    thermo_style += f" c_c{i + 1}[4]"

        f.write("\n")
        f.write(f"thermo_style    {thermo_style}\n")
        f.write("thermo_modify   lost error norm yes\n")
        f.write(f"run             {general.run}\n\n")
        f.close()

    def process(self):
        """
        Post-process temperature-ramping simulations.

        Returns
        -------
        numpy.ndarray
            Array of shape (N_T, 2) containing temperature and
            averaged enthalpy values [[T1, H1], [T2, H2], ...].
        """
        natom = self._natom
        H_of_T = []
        for T in self._Tlist:
            Tdir = f"{self._dir}/T{T:g}"
            if not os.path.isdir(Tdir):
                raise Exception(
                    f"Error: {Tdir} does not exist for post processing!"
                )
            job = lammpsJob(directory=Tdir)
            [T, H] = job.sample(
                varList=["Temp", "Enthalpy"],
                logfile="log.lammps"
            )
            H_of_T.append([T, H])
        return np.array(H_of_T)
