from jobs.lammpsJob import *
from tools.postprocess import *
import json

inp = json.load(open("input.json", "r"))
# set up general parameters
general = lammpsPara(inp["general"])

# create header for TDB file
tdb = create_tdb_header(general.system, general.mass)

# postprocess liquid jobs
tdb += process_liquid(general, inp["liquid"], write_file=False)

# postprocess solid jobs
tdb += process_solid(general, inp["solid"], write_file=False)

# write to TBD file
fout = open(f"{general.proj_dir}/{''.join(general.system)}.TDB", "wt")
fout.write(tdb)
fout.close()
