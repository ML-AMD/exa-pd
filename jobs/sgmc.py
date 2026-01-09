import numpy as np
import os
from jobs.lammpsJob import *
from tools.utils import *
from tools.logging_config import exapd_logger
import sys


class sgmc_simulator(lammpsJobGroup):
    """
    Set up LAMMPS jobs for semi-grand canonical Monte Carlo (SGMC) simulations.

    This class generates a grid of LAMMPS simulations over temperature and
    chemical potential difference (mu_B - mu_A) for solid solutions.

    Parameters
    ----------
    data_in : str
        Initial structure file in LAMMPS data format.
    Tlist : list of float
        List of temperatures to run simulations at.
    mu_list : list of float
        List of chemical potential differences (mu_B - mu_A).
    directory : str
        Path to the group directory where job subfolders are created.
    barostat : str, optional
        Barostat type for NPT. If "none", the script attempts to run without
        barostat contribution in `baro_style` (though the current script still
        writes an `npt` fix line).
    """

    def __init__(self,
                 data_in,
                 Tlist,
                 mu_list,
                 directory,
                 barostat="iso"):
        super().__init__(directory)
        self._datain = data_in
        self._Tlist = Tlist
        self._mu_list = mu_list
        self._barostat = barostat
        natom, ntyp, nab = read_lmp_data(self._datain, read_nab=True)
        self._natom = natom
        self._ntyp = ntyp
        self._nab = nab

    def setup(self, general):
        """
        Create job directories and LAMMPS input scripts for all (T, mu) points.

        Parameters
        ----------
        general : lammpsPara
            General LAMMPS parameters (units, pair potential, neighbor settings,
            masses, timestep, thermo frequency, pressure, Tdamp/Pdamp, run length).
        """
        natom = self._natom
        for T in self._Tlist:
            Tdir = f"{self._dir}/T{T:g}"
            if not os.path.isdir(Tdir):
                try:
                    os.mkdir(Tdir)
                except Exception as e:
                    exapd_logger.critical(
                        f"{e}: Cannnot create directory {Tdir}.")
            for mu in self._mu_list:
                mu_dir = f"{self._dir}/T{T:g}/mu{mu:g}"
                scriptFile = f"{mu_dir}/lmp.in"
                job = lammpsJob(directory=mu_dir,
                                scriptFile=scriptFile)
                if not os.path.exists(scriptFile):
                    self.write_script(job._script, general, T, mu)
                self._jobList.append(job)

    def write_script(self, scriptFile, general, T, mu):
        """
        Write a LAMMPS input script for a given temperature and chemical potential.

        Parameters
        ----------
        scriptFile : str
            Output path for the LAMMPS input script.
        general : lammpsPara
            General LAMMPS parameters and pair potential definition.
        T : float
            Temperature for the simulation.
        mu : float
            Chemical potential difference (mu_B - mu_A) used in the SGMC swap fix.

        Notes
        -----
        The script uses:
        - `fix npt` for temperature/pressure control (with barostat style determined
          by `self._barostat`), and
        - `fix atom/swap ... semi-grand yes ... mu 0 {mu}` to perform SGMC swaps
          between types 1 and 2.

        Thermo output includes swap acceptance counters `f_2[1]`, `f_2[2]` and
        the type populations `c_typevec[1]`, `c_typevec[2]`.
        """
        f = open(scriptFile, 'wt')
        f.write(f"#  SLI simulation for T = {T} and mu = {mu}\n")
        f.write("\n")
        f.write(f"units           {general.units}\n")
        f.write("boundary        p p p\n")
        f.write("atom_style      atomic\n")
        f.write("atom_modify     map array\n")
        f.write("\n")
        f.write(f"read_data       {self._datain}\n")
        f.write("\n")
        f.write(general.pair._cmd)
        if general.neighbor is not None:
            f.write(f"neighbor        {general.neighbor}\n")
        f.write(f"neigh_modify    {general.neigh_modify}\n")
        f.write("\n")
        if general.mass is not None:
            if isinstance(general.mass, list):
                for i in range(len(general.mass)):
                    f.write(f"mass            {i + 1} {general.mass[i]}\n")
            else:
                f.write(f"mass            * {general.mass}\n")
        f.write("\n")
        f.write(
            f"velocity        all create {T:g} {np.random.randint(1000000)} rot yes dist gaussian\n")
        if general.timestep is not None:
            f.write(f"timestep        {general.timestep}\n")
        f.write(f"thermo          {general.thermo}\n")
        f.write("\n")
        f.write("compute         typevec all count/type atom\n")
        f.write("\n")
        if self._barostat == "none":
            baro_style = ''
        elif "couple" not in self._barostat:
            baro_style = f"{self._barostat} {general.pressure} {general.pressure} {general.Pdamp}"
        else:
            baro_style = f"x {general.pressure} {general.pressure} {general.Pdamp} "\
                + f"y {general.pressure} {general.pressure} {general.Pdamp} "\
                + f"z {general.pressure} {general.pressure} {general.Pdamp} "\
                + self._barostat
        f.write(
            f"fix             1 all npt temp {T:g} {T:g} {general.Tdamp} {baro_style}\n")
        f.write(
            f"fix             2 all atom/swap 10 1 {np.random.randint(1000000)} {T:g} semi-grand yes types 1 2 mu 0 {mu:g}\n")
        f.write("\n")
        f.write(
            f"thermo_style    custom step temp etotal press vol pe f_2[1] f_2[2] c_typevec[1] c_typevec[2]\n")
        f.write("thermo_modify   lost error norm yes\n")
        f.write(
            f"dump            1 all custom {int(0.01 * general.run)} dump.atom id type xs ys zs\n")
        f.write("dump_modify     1 sort id\n")
        f.write(f"run             {general.run}\n")
        f.close()
