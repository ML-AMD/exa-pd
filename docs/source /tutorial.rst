Tutorial
========

This tutorial demonstrates how to build and run **exa-PD** on both the GPU and CPU partitions of the `Perlmutter supercomputer <https://docs.nersc.gov/systems/perlmutter/architecture/>`_ to construct the phase diagram for the Cu-Zr system.

1. Clone the Repository
------------------------

Start by cloning the `exa-PD` repository:

.. code-block:: bash

   git clone https://github.com/ML-AMD/exa-pd.git
   cd exa-pd

----

2. Install Dependencies
------------------------



----


3. Prepare the Data and LAMMPS Setup
-----------------------------------

Ensure you have a working `LAMMPS <https://www.lammps.org/>`_ installation. If you will be using a neural network potential (NNP), you will also need an interface for the NNP in LAMMPS. For example, if you use a NNP trained by DeepMD-kit, you will need to install `DeepMD-kit <https://github.com/deepmodeling/deepmd-kit>`_ and have LAMMPS compiled with USER-DP package enabled.

Next, copy the folder that contains the crystal structures of the Cu-Zr solid phases:

.. code-block:: bash

   cp -r example/phases/ ./

----

3. Prepare the Parsl Configuration
-----------------------------------

Parsl configurations must be placed inside the ``parsl_configs/`` directory so that they can be automatically discovered by exa-PD at runtime.

Start by copying the default Perlmutter configuration:

.. code-block:: bash

   cp parsl_configs/perlmutter.py parsl_configs/my_perlmutter.py

Then edit `my_perlmutter.py` and `config_registry.py`:

a. Change the registration name
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

At the top of `config_registry.py`, update `CONFIG_REGISTRY` to reflect the new config name. This value **have to match** the value you will set in your JSON config file (the ``run`` field).

.. code-block:: python

   # Before:
   CONFIG_REGISTRY = {
    "perlmutter": "parsl_configs.perlmutter.PerlmutterConfig",
   }

   # After:
   CONFIG_REGISTRY = {
    "my_perlmutter": "parsl_configs.my_perlmutter.PerlmutterConfig",
   }


b. Update each executor
~~~~~~~~~~~~~~~~~~~~~~~

The Perlmutter configuration defines **two separate executors**: one that runs on **GPU nodes** and the other on **CPU nodes**

For each executor, update the following fields in the `SlurmProvider`:

- `max_blocks`: The maximum number of Slurm jobs that Parsl is allowed to create for that provider. It is not a lifetime cap, it’s a concurrent cap. It only limits how many blocks (Slurm jobs) can exist at the same time. If one job finishes, Parsl is free to submit another job, as long as the total number of active blocks never exceeds `max_blocks`. 

- `wall_time`: It specifies the maximum run time requested for each Slurm job allocation (block). It directly maps to Slurm’s --time option.

----

5. Prepare the JSON Input File
---------------------------------------

Copy the default input file:

.. code-block:: bash

   cp configs/input.json configs/my_input.json

Edit the following fields in `my_input.json`:

- ``dir:`` Path to the root directory of the project for running the calculations. Default is the current directory.
- ``pair_coeff:`` The pair coeff associated with the pair style in LAMMPS syntax. The path to the potential file should be changed to the absolute path.
- ``ngpu:`` The number of nodes required for each GPU slurm job submitted by Parsl. Default is 1.
- ``ncpu:`` The number of nodes required for each CPU slurm job submitted by Parsl. Default is 1.
- ``gpu_account:`` The allocation or account name to be used when submitting GPU jobs (e.g., "m1234"). 
- ``cpu_account:`` The allocation or account name to be used when submitting CPU jobs (e.g., "m1234"). 
- ``gpu_exe:`` The executable command or absolute path to run LAMMPS on GPU resources.
- ``cpu_exe:`` The executable command or absolute path to run LAMMPS on CPU resources.
- ``parsl_config:`` The Parsl configuration profile that specifies how jobs are launched and resources are allocated. This value **have to match** the value in `config_registry.py` (e.g., "my_perlmutter")
- ``qos:`` The quality-of-service (QOS) flag to control scheduling priority (e.g., "regular", "premium").
- ``data_in:`` ???
- ``ref_pair_coeff:`` The pair coefficient defining the reference system. The path to the potential file should be changed to the absolute path.

----

6. Run the Workflow
---------------------

Run the full exa-PD workflow from a login node of Perlmutter:

.. code-block:: bash

   export PYTHONPATH=$(pwd):$PYTHONPATH
   python run.py --config configs/my_input.json




