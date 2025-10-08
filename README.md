# exa-pd: Exascale Accelerated Phase Diagram construction
Exa-pd is a highly parallelizable workflow for constructing multi-element phase diagrams (PDs). It uses standard sampling techniques—molecular dynamics (MD) and Monte Carlo (MC)—as implemented in the LAMMPS package, to simultaneously sample multiple phases on a fine temperature–composition mesh for free-energy calculations. The workflow uses [Parsl](https://parsl-project.org) as a global controller to manage the MD/MC jobs to achieve massive parallelization with almost ideal scalability. The resulting free energies of both liquid and solid phases (including solid solutions) are then fed to CALPHAD modeling using the PYCALPHAD package for the construction of a multi-element PD.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Register a new Parsl configuration](#register-a-new-parsl-configuration)
- [Usage](#usage)
- [Examples](#examples)

## Prerequisites
This package requires:
- python >= 3.10
- parsl >= 2025.3.24
- numpy >= 2.0
- scipy >= 1.0.0
- ase >= 3.24.0

If you use [Conda](https://docs.conda.io/en/latest/miniconda.html) to manage Python packages, you may create a conda environment to install the required packages using the `amd_env` environment yaml file we provide:
```bash
conda env create -f amd_env.yml
conda activate amd_env
```
Additionally:
- Ensure you have a working [LAMMPS](https://www.lammps.org/) installation.
- Ensure you have prepared the crystal structures for solid phases for free energy calculations in the format of Crystallographic Information File (CIF) or Vienna Ab initio Simulation Package (VASP). 
- Create a json file that specifies all the input parameters for exa-PD. See for example [configs/input.json](configs/input.json).

## Register a new Parsl configuration
We currently support the automated workflows on NERSC's Perlmutter. If you would like to run on a different computing system, you must add your own Parsl configuration following these steps:

1. Create a new file in the `parsl_configs` directory, similar to the one in [parsl_configs/perlmutter.py](parsl_configs/perlmutter.py)
2. Add a configuration class with a unique name `<my_parsl_config_name>` 
3. Modify Parsl's execution settings. More details can be found in [Parsl's official documentation](https://parsl.readthedocs.io/en/stable/userguide/configuration/execution.html)
4. Register your configuration class by including it in `CONFIG_REGISTRY` in `config_registry.py` under `parsl_configs` directory.
5. Modify your json configuration file accordingly by setting `parsl_config` in the json configuration file to `<my_parsl_config_name>`

## Usage
- Modify the json configuration file in the `configs` directory to indicate all the input parameters

- Run exa-PD with the json file:
    ```bash
    python run.py --config configs/<your_config_file>
    ```

- After all the MD jobs are finished, run run_process.py to get the free energies for all liquid and solid phases.
    ```bash
    python run_process.py
    ```

- Feed free energies to CALPHAD modeling using the PYCALPHAD package to construct the phase diagram.
       
## Examples
Follow the step-by-step tutorial to construct the phase diagram for the Cu-Zr system: [exa-PD Tutorial](https://github.com/ML-AMD/exa-pd/blob/main/docs/source%20/tutorial.rst)   

## Copyright
Copyright 2025. Iowa State University. All rights reserved. This software was produced under U.S. Government contract DE-AC02-07CH11358 for the Ames National Laboratory, which is operated by Iowa State University for the U.S. Department of Energy. The U.S. Government has rights to use, reproduce, and distribute this software. NEITHER THE GOVERNMENT NOR IOWA STATE UNIVERSITY MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. If software is modified to produce derivative works, such modified software should be clearly marked, so as not to confuse it with the version available from the Ames National Laboratory.

© 2025. Triad National Security, LLC. All rights reserved.
This program was produced under U.S. Government contract 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is operated by Triad National Security, LLC for the U.S. Department of Energy/National Nuclear Security Administration. All rights in the program are reserved by Triad National Security, LLC, and the U.S. Department of Energy/National Nuclear Security Administration. The Government is granted for itself and others acting on its behalf a nonexclusive, paid-up, irrevocable worldwide license in this material to reproduce, prepare. derivative works, distribute copies to the public, perform publicly and display publicly, and to permit others to do so.
