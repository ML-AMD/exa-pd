import numpy as np
import os
import scipy
from jobs.lammpsJob import *
from tools.utils import *
from tools.logging_config import exapd_logger


class einstein(lammpsJobGroup):
    """
    Frenkel–Ladd thermodynamic integration using an Einstein crystal reference.

    This class implements the Frenkel–Ladd method to compute the absolute
    free energy of a solid by thermodynamic integration between the
    interacting system and an Einstein crystal.

    Parameters
    ----------
    data_in : str
        Initial structure file in LAMMPS data format.
    dlbd : float
        Increment of the coupling parameter $lambda$.
    T : float
        Temperature of the simulation.
    directory : str, optional
        Path to the job group directory.
    """

    def __init__(self,
                 data_in,
                 dlbd,
                 T,
                 directory="./"):

        super().__init__(directory)
        self._datain = data_in
        self._dlbd = dlbd
        self._lbdList = np.arange(0, 1 + 0.1 * dlbd, dlbd)
        self._lbdList[0] = 0.01
        self._lbdList[-1] = 0.99
        self._T = T
        natom, ntyp, nab = read_lmp_data(self._datain, read_nab=True)
        self._natom = natom
        self._ntyp = ntyp
        self._nab = nab

    def setup(self, general, barostat, depend):
        """
        Set up thermodynamic integration jobs for the Einstein crystal path.

        Parameters
        ----------
        general : lammpsPara
            General LAMMPS parameters.
        barostat : str
            Barostat type ("iso", "tri", etc.).
        depend : list object
            Dependency specification for the jobs.
        """
        natom = self._natom
        nlbd = int(1 / self._dlbd) + 1
        dU = []  # only used in post processing
        for ilbd, lbd in enumerate(self._lbdList):
            jobdir = f"{self._dir}/{ilbd}"
            scriptFile = f"{jobdir}/lmp.in"
            job = lammpsJob(directory=jobdir,
                            scriptFile=scriptFile, arch="cpu",
                            depend=depend)
            if not os.path.exists(scriptFile):
                self.write_script(job._script, general, lbd, barostat)
            self._jobList.append(job)

    def write_script(self, scriptFile, general, lbd, barostat):
        """
        Write a LAMMPS input script for a given $lambda value.

        Parameters
        ----------
        scriptFile : str
            Path to the LAMMPS input script to be written.
        general : lammpsPara
            General LAMMPS parameters.
        lbd : float
            Coupling parameter $lambda$.
        barostat : str
            Barostat type.
        """
        f = open(scriptFile, 'wt')
        f.write(f"#  Frenkel-Ladd for solid phase {self._nab}\n\n")
        f.write(f"units           {general.units}\n")
        f.write("boundary        p p p\n")
        f.write("atom_style      atomic\n")
        f.write("atom_modify     map array\n\n")
        f.write(f"read_data       {self._datain}\n")

        if barostat == "tri":
            f.write(
                "change_box      all x final ${fxlo} ${fxhi} y final ${fylo} ${fyhi} "
                "z final ${fzlo} ${fzhi} xy final ${fxy} xz final ${fxz} "
                "yz final ${fyz} remap units box\n"
            )
        else:
            f.write(
                "change_box      all x final ${fxlo} ${fxhi} y final ${fylo} ${fyhi} "
                "z final ${fzlo} ${fzhi} remap units box\n"
            )
        f.write("\n")

        rcut = 2.5 if general.units == "lj" else 7.5
        f.write(
            hybridPair(
                lammpsPair(f"zero {rcut}", "* *"),
                general.pair,
                lbd
            )
        )
        f.write(f"compute         U0 all pair {general.pair._name}\n")

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

        f.write(f"thermo          {general.thermo}\n\n")
        f.write("fix             1 all nve\n")
        f.write("minimize        1e-6 1e-8 5000 50000\n")
        f.write("write_restart   restart.equil\n")

        f.write(
            f"fix             2 all langevin {self._T} {self._T} "
            f"$(100*dt) {np.random.randint(1000000)} zero yes\n\n"
        )

        if general.units == "lj":
            kb = 1
        elif general.units == "metal":
            kb = 8.617333262e-5

        f.write(
            f"variable        pref equal {(1 - lbd) * 3 * kb * self._T:.6f}\n"
        )

        var_U1 = "variable        U1 equal \""
        for i in range(self._ntyp):
            if self._nab[i] > 0:
                f.write(
                    f"variable        k{i + 1} equal v_pref/v_msd{i + 1}\n"
                )
                f.write(f"group           g{i + 1} type {i + 1}\n")
                f.write(
                    f"fix             {i + 3} g{i + 1} spring/self ${{k{i + 1}}}\n"
                )
                f.write(f"fix_modify      {i + 3} energy yes\n")
                var_U1 += f"f_{i + 3}/atoms/{1 - lbd} + "
        var_U1 = var_U1[:-3] + '\"\n'
        f.write(var_U1)

        f.write("\n")
        f.write("thermo_style    custom step temp etotal press c_U0 v_U1\n")
        f.write("thermo_modify   lost error norm yes\n")
        f.write("reset_timestep  0\n")
        f.write(f"run             {general.run}\n")
        f.close()

    def process(self, general, depend):
        """
        Post-process completed jobs and compute the absolute free energy.

        Parameters
        ----------
        general : lammpsPara
            General LAMMPS parameters.
        depend : list object
            Dependency specification used for MSD sampling.

        Returns
        -------
        float
            Absolute Helmholtz free energy per atom.
        """
        if general.units == "lj":
            kb = hbar = au = 1
        elif general.units == "metal":
            kb = 8.617333262e-5
            hbar = 6.582119569e-16
            au = 1.0364269190e-28

        natom = self._natom
        T = self._T

        n_msd = 0
        for n in self._nab:
            if n > 0:
                n_msd += 1

        prejob = lammpsJob(directory=depend[0])
        msd_list = prejob.sample(varList=depend[2][-n_msd:])

        idx = 0
        f_ein = 0
        for i, n in enumerate(self._nab):
            if n > 0:
                kappa = 3 * kb * T / msd_list[idx]
                mass = au * general.mass[i]
                omega = np.sqrt(kappa / mass)
                f_i = kb * T * np.log(hbar * omega / (kb * T))
                f_ein += 3 * n / natom * f_i
                idx += 1

        dU = []
        for ilbd, lbd in enumerate(self._lbdList):
            jobdir = f"{self._dir}/{ilbd}"
            if not os.path.isdir(jobdir):
                exapd_logger.critical(
                    f"{jobdir} does not exist for post processing."
                )
            job = lammpsJob(directory=jobdir)
            [U0, U1] = job.sample(varList=["c_U0", "v_U1"])
            dU.append([lbd, U0 - U1])

        dU = np.asarray(dU)
        interpolator = scipy.interpolate.PchipInterpolator(
            dU[1:, 0], dU[1:, 1], extrapolate=True
        )
        lbd_extra = np.arange(0, 1.001, 0.01)
        dU_extra = interpolator(lbd_extra)
        dF = scipy.integrate.simpson(dU_extra, lbd_extra)

        if general.pressure == 0:
            pv = 0
        else:
            xlo, xhi, ylo, yhi, zlo, zhi = prejob.sample(
                varList=depend[2][:6]
            )
            vol = (xhi - xlo) * (yhi - ylo) * (zhi - zlo)
            pv = general.pressure * vol / self._natom

        return f_ein + dF + pv
