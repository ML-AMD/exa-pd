import numpy as np
import os
from jobs.lammpsJob import *
from tools.utils import *
import sys


class pem_simulator(lammpsJobGroup):
    """
    Set up LAMMPS jobs for nucleation simulation using the
    persistent-embryo method (PEM).

    Parameters
    ----------
    data_in : str
        Initial structure file containing a crystal embryo in liquid
        in LAMMPS data format.
    Tlist : list of float
        Temperatures to run PEM at.
    Tmelt: float
        A high temperature to melt the crystal to create initial liquid
        structure.
    directory : str
        Path to the group directory where job subfolders are created.
    rcut: float
        Cut-off distance for identifying nearest neighbors.
    repeat:
        Number of independent simulations, each initiated with a different
        random seed.
    Sc:
        Cut-off value for order parameter.
    Nsc:
        Sub-critical size for terminating the spring force.
    minNeigh:
        Number of minimal neighbors in solid-phase.
    """

    def __init__(self,
                 data_in,
                 Tlist,
                 Tmelt,
                 directory,
                 rcut,
                 repeat=1,
                 Sc=0.5,
                 Nsc=600,
                 minNeigh=5)
    super().__init__(directory)
    self._datain = data_in
    self._Tlist = Tlist
    self._Tmelt = Tmelt
    self._rcut = rcut
    self._repeat = repeat
    self._Sc = Sc
    self._Nsc = Nsc
    self._minNeigh = minNeigh
    self._ntarget = ntarget
    natom, ntyp, nab = read_lmp_data(self._datain, read_nab=True)
    self._ntyp = ntyp

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
            for i in range(self._repeat):
                job_dir = f"{self._dir}/T{T:g}-run{i}"
                scriptFile = f"{job_dir}/lmp.in"
                job = lammpsJob(directory=job_dir,
                                scriptFile=scriptFile)
            if not os.path.exists(scriptFile):
                self.write_script(job._script, general, T)
            self._jobList.append(job)


def write_script(self, scriptFile, general, T):
    """
    Write a LAMMPS input script for nucleation simulations using the
    Persistent Embryo Method (PEM).

    Parameters
    ----------
    scriptFile : str
        Output path for the LAMMPS input script.
    general : lammpsPara
        General LAMMPS parameters.
    T : float
        Target temperature.
    """
    baro_style = f"iso {general.pressure} {general.pressure} {general.Pdamp}"
    f = open(scriptFile, 'wt')
    f.write("# Persistent Embryo Method nucleation simulation\n")
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

    f.write("# set constant\n")
    f.write("   # MD\n")
    f.write(f"   variable        temp equal {T}\n")
    f.write(f"   variable        htemp equal {self._Tmelt}\n")
    f.write("   variable        nloop equal 10000\n")
    f.write("   # BOO_Q6 parameter\n")
    f.write(f"   variable        rcut equal {self._rcut}\n")
    f.write(f"   variable        qq6_cut equal {self._Sc}\n")
    f.write(f"   variable        connect_cut equal {minNeigh}\n")
    f.write("   # spring constant k=kspr0*scaler, while scaler=(Nsc-Nsolid)/Nsc\n")
    f.write("   variable        kspr0 equal 100\n")
    f.write("   variable        time_window equal 3000 # *1fs, the time window to change the spring constant\n")
    f.write("   variable        dump_window equal 200 # *1fs, the time window to dump trajectory\n")
    f.write(f"   variable        Nsc equal {Nsc}\n")
    f.write(f"   variable        Nbreak equal {5 * Nsc}\n\n")

    f.write("# thermo\n")
    f.write(f"timestep        {general.timestep}\n")
    f.write("group liquid type <= 1\n")
    f.write("group seed type > 1\n")
    f.write("compute         scom seed com\n")
    f.write("thermo          {general.thermo}\n")
    f.write(
        "thermo_style    custom step temp etotal pe ke press enthalpy vol c_scom[1] c_scom[2] c_scom[3]\n\n")

    f.write("# generate initial liquid\n")
    f.write("reset_atoms id\n")
    f.write("velocity all create ${temp} ${vseed} rot yes dist gaussian\n")
    f.write(
        "fix            anneal liquid npt temp ${htemp} ${htemp} {general.Tdamp} {baro_style}\n")
    f.write("run            100000\n")
    f.write("unfix          anneal\n")
    f.write(
        "fix            cool liquid npt temp ${htemp} ${temp} {general.Tdamp} {baro_style}\n")
    f.write("run            1000\n")
    f.write("unfix          cool\n")
    f.write("\n")

    f.write("variable        scaler equal 0.01\n")
    f.write("variable        kspr equal ${kspr0}*${scaler}\n")
    f.write("# equilibrium\n")
    f.write("fix             spring seed spring/self ${kspr} xyz\n")
    f.write("fix_modify      spring energy no\n")
    f.write(
        "fix             eql all npt temp ${temp} ${temp} {general.Tdamp} {baro_style}\n")
    f.write("fix             com_seed seed recenter INIT INIT INIT shift all\n")
    f.write("run             1000\n")
    f.write("unfix           eql\n")
    f.write("unfix           com_seed\n")
    f.write("unfix           spring\n")
    f.write("\n")

    f.write("# nucleation start\n")
    f.write("reset_timestep  0\n")
    f.write("\n")

    f.write("# compute Q6.Q6*\n")
    f.write("thermo          100\n")
    f.write(
        "compute         Q6 all orientorder/atom degrees 1 6 components 6 nnn NULL cutoff ${rcut}\n")
    f.write("compute         QConn all coord/atom orientorder Q6 ${qq6_cut}\n")
    f.write("compute         sum_conn all reduce sum c_QConn\n")
    f.write("variable        q6Coor atom \"c_QConn >= v_connect_cut\"\n")
    f.write(
        "group           solid dynamic all var q6Coor every ${time_window}\n")
    f.write("compute         clustering solid cluster/atom ${rcut}\n")
    f.write("compute         cluster_index solid chunk/atom c_clustering\n")
    f.write("compute         cluster_size  solid property/chunk cluster_index count\n")
    f.write("variable        Nsolid equal max(c_cluster_size)\n")
    f.write(
        "thermo_style    custom step temp etotal pe ke press enthalpy vol c_scom[1] c_scom[2] c_scom[3] c_sum_conn v_Nsolid\n")
    f.write("run 0\n")
    f.write("print \"ok\"\n")
    f.write("\n")

    f.write("# NPT\n")
    f.write(
        "fix             ncl all npt temp  ${temp} ${temp} {general.Tdamp} {baro_style}\n")
    f.write("fix             nodrift all momentum 1 linear 1 1 1\n")
    f.write("fix             com_seed seed recenter INIT INIT INIT shift all\n")
    f.write("\n")

    f.write("# compute position\n")
    f.write("compute         uwpos all property/atom xu yu zu\n")
    f.write("\n")

    f.write("# define equal-style variables for COM components\n")
    f.write("variable        scomx equal c_scom[1]\n")
    f.write("variable        scomy equal c_scom[2]\n")
    f.write("variable        scomz equal c_scom[3]\n")
    f.write("\n")

    f.write("# now define per-atom variables relative to COM\n")
    f.write("variable        px  atom c_uwpos[1]-v_scomx\n")
    f.write("variable        py  atom c_uwpos[2]-v_scomy\n")
    f.write("variable        pz  atom c_uwpos[3]-v_scomz\n")
    f.write("\n")

    f.write("# store initial seeds' positions\n")
    f.write("fix             pseed0 seed store/state 0 v_px v_py v_pz\n")
    f.write("\n")

    f.write("variable        scaler equal 0.1\n")
    f.write("\n")

    f.write("# loop starts\n")
    f.write("label           loop_mark\n")
    f.write("   variable        j loop ${nloop}\n")
    f.write("   variable        kspr equal ${kspr0}*${scaler}\n")
    f.write("   print           \"LOOP= ${j} starts,  k= ${kspr}\"\n")
    f.write("   variable        sprx atom ${kspr}*(f_pseed0[1]-v_px)\n")
    f.write("   variable        spry atom ${kspr}*(f_pseed0[2]-v_py)\n")
    f.write("   variable        sprz atom ${kspr}*(f_pseed0[3]-v_pz)\n")
    f.write("   fix             spring seed addforce v_sprx v_spry v_sprz every 1\n")
    f.write("   thermo_style    custom step temp etotal pe ke press enthalpy vol v_Nsolid  v_kspr\n")
    f.write(
        "   dump            2 all custom ${dump_window} all.*.atom id type xs ys zs vx vy vz c_QConn\n")
    f.write("   dump_modify     2 sort id\n")
    f.write("   run             ${time_window}\n")
    f.write("\n")
    f.write("   # update spring constant according to Nsolid\n")
    f.write("   variable        ss equal (${Nsc}-${Nsolid})/${Nsc}\n")
    f.write("   variable        scaler equal (${ss}>0)*${ss}\n")
    f.write(
        "   print           \"LOOP= ${j} k= ${kspr} finished. Nsolid= ${Nsolid} Nsc= ${Nsc}\"\n")
    f.write("\n")
    f.write("   if \"${Nsolid} > ${Nbreak}\" then \"jump  SELF break\"\n")
    f.write("   undump          2\n")
    f.write("   unfix           spring\n")
    f.write("   next            j\n")
    f.write("   jump            SELF loop_mark\n")
    f.write("\n")
    f.write("label           break\n")

    f.close()
