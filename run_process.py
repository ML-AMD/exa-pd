"""
Post-processing entry-point script for ExaPD workflows.

This script performs post-processing after all LAMMPS simulations
have completed. It reads the same JSON configuration file used for
the simulation runs and generates:

- Gibbs free-energy data files for liquid and solid phases
- A CALPHAD-compatible TDB file containing all fitted thermodynamic
  parameters

The script is intended to be run *after* ``run.py`` has finished
successfully.

Example
-------
Run post-processing with the same configuration file::

    python run_process.py --config input.json
"""

from jobs.lammpsJob import *
from tools.postprocess import *
import json
from tools.logging_config import exapd_logger
from tools.config_manager import ConfigManager


if __name__ == '__main__':
    """
    Execute ExaPD post-processing workflows.

    This block:
    1. Parses the configuration file.
    2. Reconstructs general LAMMPS parameters.
    3. Post-processes liquid and solid simulations.
    4. Writes Gibbs free-energy tables.
    5. Generates and writes a CALPHAD TDB file.
    """

    # read configuration
    inp = ConfigManager()
    run_config = inp["run"]

    # set up general parameters
    try:
        general = lammpsPara(inp["general"])
    except Exception as e:
        exapd_logger.critical(f"{e}: Setting up lammpsPara failed")

    # create header for TDB file
    tdb = create_tdb_header(general.system, general.mass)

    # postprocess liquid jobs
    if "liquid" in inp:
        tdb += process_liquid(general, inp["liquid"], write_file=True)

    # postprocess solid jobs
    if "solid" in inp:
        tdb += process_solid(general, inp["solid"], write_file=True)

    # write to TDB file
    fout = open(f"{general.proj_dir}/{''.join(general.system)}.TDB", "wt")
    fout.write(tdb)
    fout.close()
