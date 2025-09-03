import numpy as np
import os
from jobs.lammpsJob import *
from tools.utils import *
import sys


class tramp(lammpsJobGroup):
    '''
    set up lammps jobs for temperature ramping for any phae (liquid or solid)
    '''

    def __init__(self,
                 data_in,          # initial structure file in lammps format
                 Tlist,            # list of temperatures to equilibrate
                 directory,        # path to group directory
                 mode="scratch",   # scratch, restart or process
                 nab=None,         # number of atoms of each type, [na, nb, ...],
                 # if given, change comp in data_in accordingly
                 barostat="iso",   # barostat for npt, if "none", run nvt
                 ):
        super().__init__(directory)
        self._datain = data_in
        self._Tlist = Tlist
        self._mode = mode
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
        '''
        if boxdims=True, output detailed box dimensions,
        otherwise only output volume.
        if msd=Ture, calculate msd for each elements.
        '''
        natom = self._natom
        H_of_T = []  # only used in post processing
        for T in self._Tlist:
            Tdir = f"{self._dir}/T{T:g}"
            scriptFile = f"{Tdir}/lmp.in"
            if self._mode == "scratch":
                job = lammpsJob(directory=Tdir,
                                scriptFile=scriptFile)
                self.write_script(job._script, general, T, boxdims, msd)
                self._jobList.append(job)
            elif self._mode == "restart":
                if not os.path.isdir(Tdir):
                    raise Exception(f"Error: {Tdir} does not exist for restart job!")
                if os.path.exists(f"{Tdir}/done"):
                    continue
                if not os.path.exists(scriptFile):
                    raise Exception(f"Error: lmp script {scriptFile} does not exist for restart job!")
                job = lammpsJob(directory=Tdir, scriptFile=scriptFile)
                self._jobList.append(job)
        return 0

    def write_script(self, scriptFile, general, T, boxdims, msd):
        f = open(scriptFile, 'wt')
        f.write(f"#  Equilibrate the structure for T = {T}\n")
        f.write("\n")
        f.write(f"units           {general.units}\n")
        f.write("boundary        p p p\n")
        f.write("atom_style      atomic\n")
        f.write("atom_modify     map array\n")
        f.write("\n")
        f.write(f"read_data       {self._datain}\n")
        f.write("\n")
        if self._resetTypes:  # change composition in data_in
            f.write(reset_types(self._nab, self._natom))
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
        f.write(f"velocity        all create {T:g} {np.random.randint(1000000)} rot yes dist gaussian\n")
        if general.timestep is not None:
            f.write(f"timestep        {general.timestep}\n")
        f.write(f"thermo          {general.thermo}\n")
        if not boxdims:
            thermo_style = "custom step temp etotal press vol enthalpy"
        elif self._barostat == "tri":
            thermo_style = "custom step temp etotal press xlo xhi ylo yhi zlo zhi xy xz yz enthalpy"
        else:
            thermo_style = "custom step temp etotal press xlo xhi ylo yhi zlo zhi enthalpy"
        if self._barostat == "none":
            baro_style = ''
        elif "couple" not in self._barostat:
            baro_style = f"{self._barostat} {general.pressure} {general.pressure} {general.Pdamp}"
        else:
            baro_style = f"x {general.pressure} {general.pressure} {general.Pdamp} "\
                + f"y {general.pressure} {general.pressure} {general.Pdamp} "\
                + f"z {general.pressure} {general.pressure} {general.Pdamp} "\
                + self._barostat
        f.write(f"fix             1 all npt temp {T:g} {T:g} {general.Tdamp} {baro_style}\n")
        if msd:
            # pre-equilibrate to account for volume change
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
        f.write(f"run             {general.run}\n")
        f.write("\n")
        f.close()

    def process(self):
        natom = self._natom
        H_of_T = []  # only used in post processing
        for T in self._Tlist:
            Tdir = f"{self._dir}/T{T:g}"
            if not os.path.isdir(Tdir):
                raise Exception(f"Error: {Tdir} does not exist for post processing!")
            # if not os.path.exists(f"{Tdir}/done"):
            #    raise Exception("Error: job is not done in {Tdir} for post processing!")
            job = lammpsJob(directory=Tdir)
            [T, H] = job.sample(varList=["Temp", "Enthalpy"],
                                logfile="log.lammps")
            H_of_T.append([T, H])
        return np.array(H_of_T)
