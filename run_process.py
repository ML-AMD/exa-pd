from lammpsJob import *
from postprocess import *
import json

inp = json.load(open("input.json", "r"))
# set up general parameters 
general = lammpsPara(inp["general"])

# postprocess liquid jobs
G0 = -4.1964
T0 = 915.7 
tdb = ''
#tdb += process_liquid(general, inp["liquid"], T0, G0, write_file=False)

# postprocess solid jobs 
tdb += process_solid(general, inp["solid"], write_file=False)
print(tdb)

