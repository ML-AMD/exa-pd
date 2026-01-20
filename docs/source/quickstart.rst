Quickstart
==========

Prerequisites
-------------
**This package requires:**

- python >= 3.10
- parsl >= 2025.3.24
- numpy >= 2.0
- scipy >= 1.0.0
- ase >= 3.24.0

**Additionally:**

- Ensure you have a working LAMMPS installation
- Ensure you have prepared the initial crystal structures in the Crystallographic Information File (CIF) format or the Vienna Ab initio Simulation Package (VASP) format

.. _installation:

Installation
------------

**Option A — From Release (preferred)**

Install the packaged wheel (activate your Conda env first if you use Conda):

.. code-block:: bash

   pip install ".../exa_pd-0.1.0-py3-none-any.whl"

Quick check:

.. code-block:: bash

   exa_pd --help


**Option B — From source (Conda-only)**

Create/activate the environment, then run from the repo:

.. code-block:: bash

   conda env create -f exa_pd_env.yml
   conda activate exa_pd_env
   # from the repository root
   python run.py --help


Using a JSON Configuration File
-------------------------------

The recommended way to configure exa-PD is through a JSON configuration file.
It specifies all the required and optional parameters for running the workflow.

Here is an example configuration file for the Perlmutter system:

.. code-block:: json

    {
        "general" : {
		        "system" : "Cu Zr",
		        "mass" : [63.546, 91.224],
		        "dir" : "./test",
		        "pair_style" : "eam/fs",
		        "pair_coeff" : "* * ./example/v10_5_CuZr_B2.eam.fs Cu Zr",
		        "run" : 50000
	          },
        "run" : {
		        "ngpu" : 2,
		        "ncpu" : 2,
		        "gpu_schedule_option" : [
			          "#SBATCH -C gpu",
			          "#SBATCH -t 01:00:00",
			          "#SBATCH -A m4802",
			          "#SBATCH --gpus-per-node=4",
			          "#SBATCH -q premium"
		              ],
		        "cpu_schedule_option" : [
                	  "#SBATCH -C cpu",
                	  "#SBATCH -t 01:00:00",
			          "#SBATCH -A m4802",
                	  "#SBATCH -q premium"
                	  ],
		        "gpu_exe" : "/path/to/lmp_serial -sf gpu -pk gpu 1",
		        "cpu_exe" : "lmp",
		        "parsl_config" : "perlmutter"
	          	},
	  "solid" : {
		        "phases" : [
			          "./phases/Cu.cif",
			          "./phases/Zr3Cu8.poscar",
			          "./phases/Zr7Cu10.cif",
			          "./phases/Zr2Cu.poscar",
			          "./phases/ZrCu.cif",
			          "./phases/Zr.poscar"
		              ],
		        "Tmin" : 300,
		        "Tmax" : 1500,
		        "dT" : 50,
		        "Tlist" : [2000],
		        "ntarget" : 4000
	          	},
	  "liquid" : {
				"data_in" : "./example/liq.lammps",
				"initial_comp" : [1, 0],
				"final_comp" : [0, 1],
				"ncomp" : 10,
				"ref_pair_style" : "eam/fs",
				"ref_pair_coeff" : "* * ./example/v10_5_CuZr_B2.eam.fs Cu Cu",
				"dlbd" : 0.05,
				"Tmin" : 700,
				"Tmax" : 2000,
				"dT" : 50
				}
    }

