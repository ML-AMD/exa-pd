# exa-pd: Exascale Accelerated Phase Diagram construction
Exa-pd is a highly parallelizable workflow for constructing multi-element phase diagrams (PDs). It uses standard sampling techniques—molecular dynamics (MD) and Monte Carlo (MC)—as implemented in the LAMMPS package, to simultaneously sample multiple phases on a fine temperature–composition mesh for free-energy calculations. The workflow uses [Parsl](https://parsl-project.org) as a global controller to manage the MD/MC jobs to achieve massive parallelization with almost ideal scalability. The resulting free energies of both liquid and solid phases (including solid solutions) are then fed to CALPHAD modeling using the PYCALPHAD package for the construction of a multi-element PDs.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Usage](#usage)
- [Register a new Parsl configuration](#register-parsl-config)
- [Examples](#examples)

## Prerequisites
This package requires:
- python >= 3.10
- parsl >= 2025.3.24
- ase >= 3.24.0
- sphinx >= 7.1.2
- sphinx_rtd_theme >= 3.0.2
- mp-api >= 0.45.7

If you use [Conda](https://docs.conda.io/en/latest/miniconda.html) to manage Python packages, you may create a conda environment to install the required packages using the `amd_env` environment yaml file we provide:
```bash
conda env create -f amd_env.yml
conda activate amd_env
```
Additionally:
- Ensure you have a working [LAMMPS](https://www.lammps.org/) installation.
- Ensure you have prepared the crystal structures for solid phases (line compounds) for free energy calculations in the format of Crystallographic Information File (CIF) or Vienna Ab initio Simulation Package (VASP). 
- Create a json file that specifies all the input parameters for exa-PD. See for example [configs/input.json](configs/input.json).


  
