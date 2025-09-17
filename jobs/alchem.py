import numpy as np
import os
import scipy
from jobs.lammpsJob import *
from tools.utils import *


class alchem(lammpsJobGroup):
    '''
    alchemical process to transfer from reference potential to final potential
    '''

    def __init__(self,
                 data_in,          # initial structure file in lammps format
                 dlbd,             # delta_lambda for thermodynamical integration
                 T,                # temperature
                 directory,        # path to group directory
                 ref_pair=None,    # reference pair style/coeff for TI
                 # number of atoms of each type, [na, nb, ...],
                 nab=None,
                 # if given, change comp in data_in accordingly
                 barostat="iso",   # barostat for npt, if "none", run nvt
                 ):
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
        '''
        set up TI jobs for alchemical path
        '''
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
        natom = self._natom
        if self._ref_pair is None:  # use UFM as reference by default
            if general.units == "lj":
                kb = 1
                sigma = 0.5  # LJ length unit
            elif general.units == "metal":
                kb = 8.617333262e-5
                sigma = 1.5  # angstroms
            pair0 = lammpsPair(
                f"ufm {5*sigma}", f"* * {kb*self._T*50} {sigma}")  # default p=50
            barostat = "none"  # run nvt

        else:
            pair0 = self._ref_pair
            barostat = self._barostat

        pair1 = general.pair
        numTyp0, coeff0 = pair0._numTyp[0], pair0._coeff[0]
        numTyp1, coeff1 = pair1._numTyp[0], pair1._coeff[0]
        f = open(scriptFile, 'wt')
        f.write(f"#  Liquid alchem for {general.system}\n")
        f.write("\n")
        f.write(f"units           {general.units}\n")
        f.write("boundary        p p p\n")
        f.write("atom_style      atomic\n")
        f.write("atom_modify     map array\n")
        f.write("\n")
        f.write(f"read_data       {self._datain}\n")
        f.write("\n")
        if self._resetTypes:
            f.write(reset_types(self._nab, natom))
        f.write("\n")
        # rescale the box for nvt if UFM is used as ref
        if self._ref_pair is None:
            f.write("variable        a equal v_vol^(1.0/3.0)\n")
            f.write(
                "change_box      all x final 0 $a y final 0 $a z final 0 $a remap units box\n")
        f.write("\n")
        f.write(hybridPair(pair0, pair1, lbd))
        if pair0._name == pair1._name:
            f.write(f"compute         U0 all pair {pair0._name} 1\n")
            f.write(f"compute         U1 all pair {pair1._name} 2\n")
        else:
            f.write(f"compute         U0 all pair {pair0._name} \n")
            f.write(f"compute         U1 all pair {pair1._name} \n")
        f.write("\n")
        if general.neighbor is not None:
            f.write(f"neighbor        {general.neighbor}\n")
        f.write(f"neigh_modify    {general.neigh_modify}\n")
        f.write("\n")
        if general.mass is not None:
            if isinstance(general.mass, list):
                for i in range(len(general.mass)):
                    f.write(f"mass            {i+1} {general.mass[i]}\n")
            else:
                f.write(f"mass            * {general.mass}\n")
        f.write("\n")
        f.write(
            f"velocity        all create {self._T:g} {np.random.randint(1000000)} rot yes dist gaussian\n")
        if general.timestep is not None:
            f.write(f"timestep        {general.timestep}\n")
        f.write("\n")
        f.write(f"thermo          {general.thermo}\n")
        f.write("thermo_style    custom step temp vol etotal c_U0 c_U1\n")
        f.write("thermo_modify   lost error norm yes\n")
        f.write("\n")
        if barostat == "none":  # run nvt
            f.write(
                f"fix             1 all nvt temp {self._T} {self._T} {general.Tdamp}\n")
        elif "couple" not in barostat:
            baro_style = f"{barostat} {general.pressure} {general.pressure} {general.Pdamp}"
            f.write(
                f"fix             1 all npt temp {self._T} {self._T} {general.Tdamp} {baro_style}\n")
        else:
            baro_style = f"x {general.pressure} {general.pressure} {general.Pdamp} "\
                + f"y {general.pressure} {general.pressure} {general.Pdamp} "\
                + f"z {general.pressure} {general.pressure} {general.Pdamp} "\
                + barostat
            f.write(
                f"fix             1 all npt temp {self._T} {self._T} {general.Tdamp} {baro_style}\n")

        f.write(f"run             {general.run}\n")
        f.close()

    def process(self, general):
        # contribution from thermodynamic integration
        dU = []
        for ilbd, lbd in self._lbdList:
            jobdir = f"{self._dir}/{ilbd}"
            if not os.path.isdir(jobdir):
                raise Exception(
                    f"Error: {jobdir} does not exist for post processing!")
            # if not os.path.exists(f"{jobdir}/DONE"):
            #    raise Exception(f"Error: job is not DONE in {jobdir} for post processing!")
            job = lammpsJob(directory=jobdir)
            [U0, U1] = job.sample(varList=["c_U0", "c_U1"],
                                  logfile="log.lammps")
            dU.append([lbd, U1 - U0])
        dU = np.asarray(dU)
        dG = scipy.integrate.simpson(dU[:, 1], dU[:, 0])
        if general.units == "lj":
            kb = 1
        elif general.units == "metal":
            kb = 8.617333262e-5  # eV / K
        for i in range(self._ntyp):
            if self._nab[i] > 0:
                xi = self._nab[i] / self._natom
                # contribution from mixing entropy
                dG += kb * self._T * xi * np.log(xi)
                # contribution from mass change
                dG += 1.5 * kb * self._T * xi * \
                    np.log(general.mass[0] / general.mass[i])
        return dG
