from jobs.lammpsJob import *
from tools.postprocess import *
import json
from tools.logging_config import exapd_logger
from tools.config_manager import ConfigManager

if __name__ == '__main__':
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

    # write to TBD file
    fout = open(f"{general.proj_dir}/{''.join(general.system)}.TDB", "wt")
    fout.write(tdb)
    fout.close()
