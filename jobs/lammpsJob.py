import numpy as np
from tools.logging_config import exapd_logger
import sys
import os


class lammpsJob:
    """
    Define a LAMMPS job with an input script and all necessary data files.

    Parameters
    ----------
    directory : str
        Directory where the job will be executed.
    scriptFile : str, optional
        LAMMPS input script file.
    arch : str, optional
        Architecture type (e.g., "gpu", "cpu"). Default is "gpu".
    depend : list object, optional
        Dependency information for the job.
        depend[0]: pre-job directory.
        depend[1]: list of variable names to be set in LAMMPS script.
        depend[2]: corresponding list ofvariable names in pre-job output.
    priority : int, optional
        Job priority:
        - 0: pre_job
        - 1: regular job
        - 2: dependent job
    """

    def __init__(self,
                 directory,
                 scriptFile=None,
                 arch="gpu",
                 depend=None,
                 priority=1):
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
            except Exception as e:
                exapd_logger.critical(
                    f"{e}: Cannot create directory {directory}.")
        self._script = scriptFile

    def get_dir(self):
        """
        Return the job directory.

        Returns
        -------
        str
            Job directory path.
        """
        return self._dir

    def get_script(self):
        """
        Return the LAMMPS input script file.

        Returns
        -------
        str or None
            Input script filename.
        """
        return self._script

    def get_arch(self):
        """
        Return the architecture type.

        Returns
        -------
        str
            Architecture identifier.
        """
        return self._arch

    def get_depend(self):
        """
        Return dependency information.

        Returns
        -------
        list or None
            Dependency list.
        """
        return self._depend

    def set_arch(self, arch):
        """
        Set the architecture type.

        Parameters
        ----------
        arch : str
            Architecture identifier.
        """
        self._arch = arch

    def sample(self, varList=["PotEng"], logfile="log.lammps", skip=0.2):
        """
        Sample averaged quantities from a LAMMPS log file.

        Parameters
        ----------
        varList : list of str, optional
            Names of variables to sample (e.g., "PotEng", "Temp").
        logfile : str, optional
            Name of the LAMMPS log file.
        skip : float, optional
            Fraction of initial data to discard as equilibration.

        Returns
        -------
        numpy.ndarray
            Averaged values of the requested variables.

        Notes
        -----
        This method assumes standard LAMMPS thermo output format.
        """
        outfile = f"{self._dir}/{logfile}"
        if not os.path.exists(outfile):
            exapd_logger.critical(f"{outfile} does not exist!")
        with open(outfile) as infile:
            for i, line in enumerate(infile):
                if 'Step ' in line:
                    beginline = i + 1
                    varLogged = line.split()
                if 'Loop ' in line:
                    endline = i
        cols = []
        for var in varList:
            try:
                cols.append(varLogged.index(var))
            except ValueError:
                exapd_logger.critical(f"{var} is not sampled in {outfile}!")
        data = np.loadtxt(outfile,
                          skiprows=beginline,
                          max_rows=endline - beginline,
                          usecols=cols,
                          ndmin=2)
        return np.mean(data[int(skip * len(data)):, :], axis=0)


class lammpsJobGroup:
    """
    Define a group of LAMMPS jobs to be launched simultaneously.

    Parameters
    ----------
    directory : str, optional
        Path to the group directory.
    """

    def __init__(self, directory="./"):
        from typing import List
        self._dir = directory
        if not os.path.isdir(directory):
            try:
                os.mkdir(directory)
            except Exception as e:
                exapd_logger.critical(
                    f"{e}: Cannot create directory {directory}!")
        self._jobList: List[lammpsJob] = []

    def get_joblist(self):
        """
        Return the list of LAMMPS jobs.

        Returns
        -------
        list of lammpsJob
            Job list.
        """
        return self._jobList


class lammpsPair:
    """
    Define an interatomic potential for LAMMPS.

    Parameters
    ----------
    pair_style : str
        LAMMPS pair_style command.
    pair_coeff : str or list of str
        LAMMPS pair_coeff command(s).
    """

    def __init__(self, pair_style, pair_coeff):
        values = pair_style.split()
        self._name = values[0]
        if len(values) > 1:
            for i in range(1, len(values)):
                if os.path.isfile(values[i]):
                    values[i] = os.path.abspath(values[i])
            self._param = ' '.join(values[1:])
        else:
            self._param = ''
        self._cmd = f"pair_style\t{self._name} {self._param}\n"
        if isinstance(pair_coeff, list):
            values = pair_coeff
        else:
            values = [pair_coeff]
        self._numTyp = []
        self._coeff = []
        for line in values:
            words = line.split()
            self._numTyp.append(' '.join(words[:2]))
            if len(words) > 2:
                for i in range(2, len(words)):
                    if os.path.isfile(words[i]):
                        words[i] = os.path.abspath(words[i])
                self._coeff.append(' '.join(words[2:]))
            else:
                self._coeff.append('')
            self._cmd += f"pair_coeff\t{self._numTyp[-1]} {self._coeff[-1]}\n"


class lammpsPara:
    """
    Define general LAMMPS parameters from a dictionary.

    Parameters
    ----------
    general : dict
        Dictionary containing LAMMPS parameters.
    """

    def __init__(self, general):
        self.system = general["system"].split()
        self.mass = general.get("mass", None)
        if isinstance(self.mass, list) and len(self.mass) != len(self.system):
            exapd_logger.critical(
                "error: number of elements in mass and system doesn't match!"
            )

        self.units = general.get("units", "metal")

        pair_style = general["pair_style"]
        pair_coeff = general["pair_coeff"]
        self.pair = lammpsPair(pair_style, pair_coeff)
        try:
            self.proj_dir = os.path.abspath(general["dir"])
        except KeyError:
            self.proj_dir = os.getcwd()
        if not os.path.isdir(self.proj_dir):
            try:
                os.mkdir(self.proj_dir)
            except Exception as e:
                exapd_logger.critical(
                    f"{e}: Cannot create directory {self.proj_dir}!")
        self.neighbor = general.get("neighbor", None)
        self.neigh_modify = general.get("neigh_modify", "delay 10")
        self.timestep = general.get("timestep", None)
        self.thermo = general.get("thermo", 100)
        self.pressure = general.get("pressure", 0)
        self.Tdamp = general.get("Tdamp", "$(100.0*dt)")
        self.Pdamp = general.get("Pdamp", "$(1000.0*dt)")
        self.run = general.get("run", 1000000)


def hybridPair(pair0, pair1, lbd):
    """
    Create a hybrid scaled pair style.

    Parameters
    ----------
    pair0 : lammpsPair
        First potential.
    pair1 : lammpsPair
        Second potential.
    lbd : float
        Mixing parameter (0 ≤ lbd ≤ 1).

    Returns
    -------
    str
        LAMMPS input script string.
    """
    cmd = f"pair_style\thybrid/scaled {
        1 -
        lbd} {
        pair0._name} {
            pair0._param} {lbd} {
                pair1._name} {
                    pair1._param}\n"
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
    """
    Reset atomic types based on desired composition.

    Parameters
    ----------
    nab : list of int
        Number of atoms of each type.
    natom : int
        Total number of atoms.

    Returns
    -------
    str
        LAMMPS commands to reset atom types.
    """
    if sum(nab) != natom or min(nab) < 0:
        exapd_logger.critical("Error: Incorrect number of atoms!")
    ntot = nab[0]
    cmd = f"group           g1 id <= {ntot}\n"
    cmd += f"set             group g1 type 1\n"
    for i in range(1, len(nab)):
        if nab[i] > 0:
            cmd += f"group           g{i +
                                       1} id <> {ntot +
                                                 1} {ntot +
                                                     nab[i]}\n"
            cmd += f"set             group g{i + 1} type {i + 1}\n"
            ntot += nab[i]
    return cmd


def pre_process(depend):
    """
    Pre-process a dependent job by sampling parameters from a pre-job.

    Parameters
    ----------
    depend : list
        Dependency specification:
        - depend[0]: directory of the pre-job
        - depend[1]: variable names to be set
        - depend[2]: variables to sample

    Returns
    -------
    str
        Command-line arguments for LAMMPS variables.
    """
    job = lammpsJob(directory=depend[0])
    res = job.sample(varList=depend[2])
    run_para = ''
    for var, value in zip(depend[1], res):
        run_para += f"-v {var} {value:.6f} "
    return run_para
