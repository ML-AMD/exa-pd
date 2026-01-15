import numpy as np
import os
import scipy
from jobs.lammpsJob import *
from tools.utils import *
from tools.ufgenerator import get_UF
from tools.logging_config import exapd_logger


class alchem(lammpsJobGroup):
    """
    Alchemical thermodynamic integration (TI) workflow using LAMMPS.

    This class constructs and manages a set of LAMMPS jobs along an
    alchemical path that interpolates between a reference potential
    and a target potential using a coupling parameter $lambda$.

    Parameters
    ----------
    data_in : str
        Initial liquid structure file in LAMMPS data format.
    dlbd : float
        Increment of the alchemical coupling parameter $lambda$.
    T : float
        Temperature of the simulation.
    directory : str
        Path to the job group directory.
    ref_pair : lammpsPair, optional
        Reference pair potential for thermodynamic integration.
        If None, a UFM reference is used.
    nab : list of int, optional
        Number of atoms of each type [n1, n2, ...]. If provided,
        atomic types in `data_in` will be reassigned.
    barostat : str, optional
        Barostat type for NPT simulations. If set to `"none"`,
        NVT simulations are used.
    """

    def __init__(self,
                 data_in,
                 dlbd,
                 T,
                 directory,
                 ref_pair=None,
                 nab=None,
                 barostat="iso"):
        super().__init__(directory)
        self._datain = data_in
        self._dlbd = dlbd
        self._lbdList = np.arange(0, 1 + 0.1 * dlbd, dlbd)
        self._T = T
        self._nab = nab
        self._barostat = barostat
        self._ref_pair = ref_pair
        if nab is not None:
            natom, ntyp = read_lmp_data(self._datain)
            self._resetTypes = True
        else:
            natom, ntyp, nab = read_lmp_data(self._datain, read_nab=True)
            self._resetTypes = False
        self._natom = natom
        self._ntyp = ntyp
        self._nab = nab

    def setup(self, general, depend):
        """
        Set up LAMMPS jobs along the alchemical path.

        Parameters
        ----------
        general : lammpsPara
            General LAMMPS parameters and potential definitions.
        depend : list object
            Dependency specification for the jobs.
        """
        natom = self._natom
        for ilbd, lbd in enumerate(self._lbdList):
            jobdir = f"{self._dir}/{ilbd}"
            scriptFile = f"{jobdir}/lmp.in"
            job = lammpsJob(directory=jobdir,
                            scriptFile=scriptFile, arch="cpu", depend=depend)
            if not os.path.exists(job._script):
                self.write_script(job._script, general, lbd)
            self._jobList.append(job)

    def write_script(self, scriptFile, general, lbd):
        """
        Write a LAMMPS input script for a given $lambda$ value.

        Parameters
        ----------
        scriptFile : str
            Path to the LAMMPS input script to be written.
        general : lammpsPara
            General LAMMPS parameters.
        lbd : float
            Alchemical coupling parameter λ.
        """
        natom = self._natom
        if self._ref_pair is None:
            if general.units == "lj":
                kb = 1
                sigma = 0.5
            elif general.units == "metal":
                kb = 8.617333262e-5
                sigma = 1.5
            pair0 = lammpsPair(f"ufm {5 * sigma}", f"* * {kb * self._T * 50} {sigma}")
            barostat = "none"
        else:
            pair0 = self._ref_pair
            barostat = self._barostat

        pair1 = general.pair
        f = open(scriptFile, 'wt')
        f.write(f"#  Liquid alchem for {general.system}\n\n")
        f.write(f"units           {general.units}\n")
        f.write("boundary        p p p\n")
        f.write("atom_style      atomic\n")
        f.write("atom_modify     map array\n\n")
        f.write(f"read_data       {self._datain}\n\n")

        if self._resetTypes:
            f.write(reset_types(self._nab, natom))
        f.write("\n")

        if self._ref_pair is None:
            f.write("variable        a equal v_vol^(1.0/3.0)\n")
            f.write(
                "change_box      all x final 0 $a y final 0 $a z final 0 $a remap units box\n"
            )
        f.write("\n")

        f.write(hybridPair(pair0, pair1, lbd))

        if pair0._name == pair1._name:
            f.write(f"compute         U0 all pair {pair0._name} 1\n")
            f.write(f"compute         U1 all pair {pair1._name} 2\n")
        else:
            f.write(f"compute         U0 all pair {pair0._name}\n")
            f.write(f"compute         U1 all pair {pair1._name}\n")

        f.write("\n")
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
            f"velocity        all create {self._T:g} "
            f"{np.random.randint(1000000)} rot yes dist gaussian\n"
        )
        if general.timestep is not None:
            f.write(f"timestep        {general.timestep}\n")
        f.write("\n")

        f.write(f"thermo          {general.thermo}\n")
        f.write("thermo_style    custom step temp vol etotal c_U0 c_U1\n")
        f.write("thermo_modify   lost error norm yes\n\n")

        if barostat == "none":
            f.write(
                f"fix             1 all nvt temp "
                f"{self._T} {self._T} {general.Tdamp}\n"
            )
        elif "couple" not in barostat:
            baro_style = (
                f"{barostat} {general.pressure} "
                f"{general.pressure} {general.Pdamp}"
            )
            f.write(
                f"fix             1 all npt temp "
                f"{self._T} {self._T} {general.Tdamp} {baro_style}\n"
            )
        else:
            baro_style = (
                f"x {general.pressure} {general.pressure} {general.Pdamp} "
                f"y {general.pressure} {general.pressure} {general.Pdamp} "
                f"z {general.pressure} {general.pressure} {general.Pdamp} "
                + barostat
            )
            f.write(
                f"fix             1 all npt temp "
                f"{self._T} {self._T} {general.Tdamp} {baro_style}\n"
            )

        f.write(f"run             {general.run}\n")
        f.close()

    def process(self, general):
        """
        Post-process completed TI jobs and compute the free energy change.

        Parameters
        ----------
        general : lammpsPara
            General LAMMPS parameters.

        Returns
        -------
        float
            Gibbs free energy difference.
        """
        if general.units == "lj":
            kb = hbar = au = 1
        elif general.units == "metal":
            kb = 8.617333262e-5
            hbar = 6.582119569e-16
            au = 1.0364269190e-28

        dU = []
        for ilbd, lbd in enumerate(self._lbdList):
            jobdir = f"{self._dir}/{ilbd}"
            if not os.path.isdir(jobdir):
                exapd_logger.critical(
                    f"{jobdir} does not exist for post processing."
                )
            if not os.path.exists(f"{jobdir}/DONE"):
                exapd_logger.warning(f"DONE does not exist in {jobdir}.")
            job = lammpsJob(directory=jobdir)
            [U0, U1] = job.sample(
                varList=["c_U0", "c_U1"],
                logfile="log.lammps"
            )
            dU.append([lbd, U1 - U0])

        dU = np.asarray(dU)
        dG = scipy.integrate.simpson(dU[:, 1], dU[:, 0])

        comp = [n / self._natom for n in self._nab]

        if self._ref_pair is None:
            if general.units == "lj":
                sigma = 0.5
            elif general.units == "metal":
                sigma = 1.5
            p = 50
            job = lammpsJob(directory=f"{self._dir}/0")
            vol = job.sample(["Volume"])[0]
            rho = self._natom / vol
            F_ig = F_idealgas(
                self._T, rho, self._natom,
                general.mass, comp, (kb, hbar, au)
            )
            x = (0.5 * (np.pi * sigma * sigma) ** 1.5) * rho
            press, F0 = get_UF(p, x)
            F0 *= (kb * self._T)
            dG += (F_ig + F0 + general.pressure / rho)
        else:
            m0 = general.mass[0]
            for x, m in zip(comp, general.mass):
                if x > 0:
                    dG += kb * self._T * x * np.log(x)
                    dG += 1.5 * kb * self._T * x * np.log(m0 / m)

        return dG
