import numpy as np
import os
from jobs.lammpsJob import *
from tools.utils import *
import sys


class sli_simulator(lammpsJobGroup):
    """
    Set up LAMMPS jobs for solid–liquid interface (SLI) simulations.

    This workflow prepares and runs a sequence of MD stages to create a
    solid–liquid interface:
    1) pre-equilibrate whole system at target temperature,
    2) melt half of the simulation box by heating only the "liquid" region,
    3) cool the melted region back to the target temperature,
    4) re-equilibrate using a uniaxial barostat along the chosen orientation.

    Parameters
    ----------
    data_in : str
        Initial structure file in LAMMPS data format.
    Tlist : list of float
        Temperatures to run the interface preparation/equilibration at.
    Tmelt : float
        High temperature used to melt half of the box.
    directory : str
        Path to the group directory where job subfolders are created.
    replicate : int, optional
        Replication factor along the interface orientation direction.
        Default is 2.
    orient : {"x", "y", "z"}, optional
        Direction along which the system is replicated and along which the
        liquid-half region is defined. Default is "z".
    barostat : str, optional
        Barostat style for the pre-equilibration stage. If "none", the script
        writes an NPT fix with an empty barostat style string (current behavior).
        Default is "iso".
    """

    def __init__(self,
                 data_in,
                 Tlist,
                 Tmelt,
                 directory,
                 replicate=2,
                 orient="z",
                 barostat="iso"):
        super().__init__(directory)
        self._datain = data_in
        self._Tlist = Tlist
        self._Tmelt = Tmelt
        self._barostat = barostat
        self._replic = replicate
        self._orient = orient
        natom, ntyp, nab = read_lmp_data(self._datain, read_nab=True)
        self._natom = natom * replicate
        self._ntyp = ntyp
        self._nab = nab * replicate

    def setup(self, general):
        """
        Create job directories and LAMMPS input scripts for all temperatures.

        Parameters
        ----------
        general : lammpsPara
            General LAMMPS parameters (units, pair potential, neighbor settings,
            masses, timestep, thermo frequency, pressure, Tdamp/Pdamp, run length).
        """
        natom = self._natom
        for T in self._Tlist:
            Tdir = f"{self._dir}/T{T:g}"
            scriptFile = f"{Tdir}/lmp.in"
            job = lammpsJob(directory=Tdir,
                            scriptFile=scriptFile)
            if not os.path.exists(scriptFile):
                self.write_script(job._script, general, T)
            self._jobList.append(job)

    def write_script(self, scriptFile, general, T):
        """
        Write a LAMMPS input script for solid–liquid interface preparation.

        Parameters
        ----------
        scriptFile : str
            Output path for the LAMMPS input script.
        general : lammpsPara
            General LAMMPS parameters and pair potential definition.
        T : float
            Target temperature for the interface simulation.

        Notes
        -----
        The script defines a `liquid` group as half of the simulation box
        along the chosen orientation (`self._orient`). It then runs several
        stages:
        - Pre-equilibrate (whole box) using NPT at temperature `T`.
        - Heat only the `liquid` group from `T` to `Tmelt` using NVT.
        - Cool the `liquid` group back from `Tmelt` to `T` using NVT.
        - Reset velocities.
        - Final relaxation using uniaxial NPT along `self._orient`.
        """
        f = open(scriptFile, 'wt')
        f.write(f"#  SLI simulation for T = {T}\n")
        f.write("\n")
        f.write(f"units           {general.units}\n")
        f.write("boundary        p p p\n")
        f.write("atom_style      atomic\n")
        f.write("atom_modify     map array\n")
        f.write("\n")
        f.write(f"read_data       {self._datain}\n")
        f.write("\n")
        if self._orient == "x":
            f.write(f"replicate       {self._replic} 1 1\n")
            f.write("variable        xmid equal (xlo+xhi)/2\n")
            f.write(
                "region          liqhalf block ${xmid} $(xhi) INF INF INF INF\n")
        elif self._orient == "y":
            f.write(f"replicate       1 {self._replic} 1\n")
            f.write("variable        ymid equal (ylo+yhi)/2\n")
            f.write(
                "region          liqhalf block INF INF ${ymid} $(yhi) INF INF\n")
        else:
            f.write(f"replicate       1 1 {self._replic}\n")
            f.write("variable        zmid equal (zlo+zhi)/2\n")
            f.write(
                "region          liqhalf block INF INF INF INF ${zmid} $(zhi)\n")
        f.write("group           liquid region liqhalf\n")
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
        f.write(f"thermo_style    custom step temp etotal press vol pe\n")
        f.write("thermo_modify   lost error norm yes\n")
        f.write("\n")
        f.write(
            f"dump            1 all custom {int(0.01 * general.run)} dump.atom id type xs ys zs\n")
        f.write("dump_modify     1 sort id\n")
        if self._barostat == "none":
            baro_style = ''
        elif "couple" not in self._barostat:
            baro_style = f"{
                self._barostat} {
                general.pressure} {
                general.pressure} {
                general.Pdamp}"
        else:
            baro_style = f"x {general.pressure} {general.pressure} {general.Pdamp} "\
                + f"y {general.pressure} {general.pressure} {general.Pdamp} "\
                + f"z {general.pressure} {general.pressure} {general.Pdamp} "\
                + self._barostat
        f.write("\n")
        f.write("# pre-equilibrate\n")
        f.write(
            f"fix             1 all npt temp {T:g} {T:g} {general.Tdamp} {baro_style}\n")
        f.write(f"run             {int(0.1 * general.run)}\n")
        f.write("unfix           1\n")
        f.write("\n")
        f.write("# melt the liquid side\n")
        f.write(
            f"fix             2 liquid nvt temp {T:g} {self._Tmelt:g} {general.Tdamp}\n")
        f.write(f"run             {int(0.5 * general.run)}\n")
        f.write("unfix           2\n")
        f.write("\n")
        f.write("# cool the liquid side\n")
        f.write(
            f"fix             3 liquid nvt temp {self._Tmelt:g} {T:g} {general.Tdamp}\n")
        f.write(f"run             {int(0.005 * general.run)}\n")
        f.write("unfix           3\n")
        f.write("\n")
        f.write("# reset velocity to remove drifting\n")
        f.write(
            f"velocity        all create {T:g} {np.random.randint(1000000)} rot yes dist gaussian\n")
        f.write("\n")
        f.write("# release the whole system using uni-axial barostat\n")
        f.write(
            f"fix             4 all npt temp {T:g} {T:g} {general.Tdamp} {self._orient} {general.pressure} {general.pressure} {general.Pdamp}\n")
        f.write(f"run             {general.run}\n")
        f.close()
