import numpy as np
import os
from jobs.lammpsJob import *
from tools.utils import *
from tools.logging_config import exapd_logger
import sys


class sgmc_simulator(lammpsJobGroup):
    '''
    set up lammps jobs for sgmc simulations for solid solutions)
    '''

    def __init__(self,
                 data_in,          # initial structure file in lammps format
                 Tlist,            # list of temperatures to equilibrate
                 mu_list,          # list of mu_B-mu_A
                 directory,        # path to group directory
                 barostat="iso",   # barostat for npt, if "none", run nvt
                 ):
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
