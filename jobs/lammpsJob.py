import os
import numpy as np
import sys


class lammpsJob:
    '''
    define a lammps job with an input script and all necessary data files.
    '''

    def __init__(self,
                 directory,           # job directory
                 scriptFile=None,     # lammps input script file
                 arch="gpu",          # whether job supports gpu acceleration
                 depend=None,         # whether job depends on other jobs
                 priority=1,          # 0: pre_job; 1: reg_job; 2: dep_job
                 ):
        self._dir = directory
        self._arch = arch
        self._depend = depend
        if depend:
            self._priority = 2
        else:
            self._priority = priority
        if not os.path.isdir(directory):
            try:
                os.mkdir(directory)
            except BaseException:
                print(f"Error: cannot create directory {directory}!")
                sys.exit(1)
        self._script = scriptFile

    def get_dir(self):
        return self._dir

    def get_script(self):
        return self._script

    def get_arch(self):
        return self._arch

    def get_depend(self):
        return self._depend

    def sample(self, varList=["PotEng"], logfile="log.lammps", skip=0.2):
        '''
        sample the average for a list of variables from LAMMPS output
        varName: name of the quantity to be sampled, pe, ke, Volume ...
        logfile: output file, default is log.lammps
        skip: fraction of lines to be skipped for establishing an equilibrium
        '''
        outfile = f"{self._dir}/{logfile}"
        if not os.path.exists(outfile):
            raise Exception(f"{outfile} does not exist!")
        with open(outfile) as infile:
            for i, line in enumerate(infile):
                if 'Step ' in line:
                    beginline = i + 1
                    varLogged = line.split()  # all logged variables
                if 'Loop ' in line:
                    endline = i
        cols = []
        for var in varList:
            try:
                cols.append(varLogged.index(var))
            except ValueError:
                print(f"{var} is not sampled in {outfile}!")
                sys.exit(1)
        data = np.loadtxt(outfile, skiprows=beginline, max_rows=endline - beginline,
                          usecols=cols, ndmin=2)
        # if data.shape[0] <= skip:
        #    skip = 0
        #    print("Warning: not enough data to be skipped, reset skip = 0.")
        return np.mean(data[int(skip * len(data)):, :], axis=0)


class lammpsJobGroup:
    '''
    define a group (list) of lammps jobs to be launched simultaneously.
    '''

    def __init__(self,
                 directory="./",       # path to group directory
                 ):
        from typing import List
        self._dir = directory
        if not os.path.isdir(directory):
            try:
                os.mkdir(directory)
            except BaseException:
                print(f"Error: cannot create directory {directory}!")
                sys.exit(1)
        self._jobList: List[lammpsJob] = []

 #  def launch(self):
 #       for job in self._jobList:
    def get_joblist(self):
        return self._jobList


class lammpsPair:
    '''
    define the interatomic potential to be used in Lammps.
    '''

    def __init__(self, pair_style, pair_coeff):
        values = pair_style.split()
        self._cmd = "pair_style\t" + pair_style + '\n'
        self._name = values[0]  # name of the pair style
        if len(values) > 1:
            # other parameters for the pair style
            self._param = ' '.join(values[1:])
        else:
            self._param = ''
        if isinstance(pair_coeff, list):
            values = pair_coeff
        else:
            values = [pair_coeff]
        self._numTyp = []  # list of numeric types for each pair_coeff
        self._coeff = []  # list of coeff for each pair_coeff
        for line in values:
            self._cmd += 'pair_coeff\t' + line + '\n'
            words = line.split()
            self._numTyp.append(' '.join(words[:2]))
            if len(words) > 2:
                self._coeff.append(' '.join(words[2:]))
            else:
                self._coeff.append('')


class lammpsPara:
    '''
    define general parameters for lammps from the general
    dictionary to be used throughout the calculations.
    '''

    def __init__(self, general):
        self.system = general["system"].split()
        try:
            self.mass = general["mass"]
            if isinstance(self.mass, list) and len(self.mass) != len(self.system):
                print("Error: number of elements in mass and system doesn't match!")
                sys.exit(1)
        except BaseException:
            self.mass = None  # has to be provided in data.in or potential file
        try:
            self.units = general["units"]
        except BaseException:
            self.units = "metal"  # default units is metal
        pair_style = general["pair_style"]
        pair_coeff = general["pair_coeff"]
        self.pair = lammpsPair(pair_style, pair_coeff)
        try:
            self.proj_dir = os.path.abspath(general["dir"])
        except BaseException:
            self.proj_dir = os.getcwd()  # default is current directory
        if not os.path.isdir(self.proj_dir):
            try:
                os.mkdir(self.proj_dir)
            except BaseException:
                print(f"Error: cannot create directory {directory}!")
                sys.exit(1)
        try:
            self.neighbor = general["neighbor"]
        except BaseException:
            self.neighbor = None
        try:
            self.neigh_modify = general["neigh_modify"]
        except BaseException:
            self.neigh_modify = "delay 10"
        try:
            self.timestep = general["timestep"]
        except BaseException:
            self.timestep = None
        try:
            self.thermo = general["thermo"]
        except BaseException:
            self.thermo = 100
        try:
            self.pressure = general["pressure"]
        except BaseException:
            self.pressure = 0
        try:
            self.Tdamp = general["Tdamp"]
        except BaseException:
            self.Tdamp = "$(100.0*dt)"
        try:
            self.Pdamp = general["Pdamp"]
        except BaseException:
            self.Pdamp = "$(1000.0*dt)"
        try:
            self.run = general["run"]
        except BaseException:
            self.run = 1000000


def hybridPair(pair0, pair1, lbd):
    '''
    create a hybrid pair style = (1-lbd)*pair0 + lbd*pair1.
    input:
         pair0, pair1: lammpsPair objects
         lbd: value between 0 and 1
    output:
          string to be included in Lammps script
    '''
    cmd = f"pair_style\thybrid/scaled {1-lbd} {pair0._name} {pair0._param} {lbd} {pair1._name} {pair1._param}\n"
    if pair0._name == pair1._name:
        name0 = pair0._name + " 1"
        name1 = pair1._name + " 2"
    else:
        name0 = pair0._name
        name1 = pair1._name
    for numTyp, coeff in zip(pair0._numTyp, pair0._coeff):
        cmd += f"pair_coeff\t{numTyp} {name0} {coeff}\n"
    for numTyp, coeff in zip(pair1._numTyp, pair1._coeff):
        cmd += f"pair_coeff\t{numTyp} {name1} {coeff}\n"
    return cmd


def reset_types(nab, natom):
    '''
    reset types for the atoms, useful to change composition from input file.
    nab: number of atoms of each type, [na, nb, ...].
    natom: total number of atoms.
    '''
    if sum(nab) != natom or min(nab) < 0:
        raise Exception("Error: Incorrect number of atoms!")
    ntot = nab[0]
    cmd = f"group           g1 id <= {ntot}\n"
    cmd += f"set             group g1 type 1\n"
    for i in range(1, len(nab)):
        if nab[i] > 0:
            cmd += f"group           g{i+1} id <> {ntot+1} {ntot+nab[i]}\n"
            cmd += f"set             group g{i+1} type {i+1}\n"
            ntot += nab[i]
    return cmd


def pre_process(depend):
    '''
    pre_process a pre-equilibrate job to get run-time params
    for a dependent job.
    depend[0]: directory for the pre-job
    depend[1]: params in lammps script to be determined
    depend[2]: values of params sampled in the pre-job
    '''
    job = lammpsJob(directory=depend[0])
    res = job.sample(varList=depend[2])
    run_para = ''
    for var, value in zip(depend[1], res):
        run_para += f"-v {var} {value:.6f} "
    return run_para
