# import bot_tools


# cmd = """c9.tool = gripper'
# proc.find_rack_index('vial', 'polymer_A')
# c9.position = loca.vial_rack[vial_indexs]
# c9.set_output('gripper', True)
# c9.position = loca.clamp
# c9.set_output('clamp', True)
# c9.set_output('gripper', False)
# """

# output = bot_tools.exec_polybot_lint_tool(cmd)
# print(output)

import loca
import pandas as pd
import robotics as ro
from robotics import procedure as proc

# Initialize hardware modules
c9 = ro.system.init('controller')
# t8 = ro.system.init('temperature')
coater = ro.system.init('coater')

# Check if polymer A is available
if 'polymer_A' not in ro.runtime['rack_status']['vial'].values:
    raise ValueError('Experiment cannot be initiated. Polymer A is not available.')

# Pick up the bernoulli substrate gripper tool
c9.tool = 'substrate_tool'

# Pick up an available substrate from the substrate rack
substrate_index = proc.find_rack_index('substrate', 'new')
c9.position = loca._substrate_rack[substrate_index]

# Activate the vacuum on the bernoulli gripper to pick up substrates
c9.set_output('substrate_tool', True)

# Move the substrate to the slide_coater and release the bernoulli gripper
c9.position = loca.s_coater
c9.set_output('substrate_tool', False)

# Pick up the polymer from the vials rack and move it to the clamp
vial_index = proc.find_rack_index('vial', 'polymer_A')
c9.position = loca.vial_rack[vial_index]

# Close the clamp, and then open the gripper
c9.set_output('clamp', True)
c9.set_output('gripper', False)

# Uncap the clamp and aspirate the polymer with the pipette
c9.uncap(pitch=1.75, revs=3.0, vel=5000, accel=5000)
c9.aspirate_ml(0, 0.5)

# Dropcast the polymer to the substrate and blade coat it
c9.dispense_ml(0, 0.2)
coater.position = 45
coater.velocity = 1
coater.position = 75

# Return the vial to the rack
c9.position = loca.clamp
c9.position = loca.vial_rack[vial_index]

# End of experiment
c9.position = loca.s_coater
c9.position = loca.substrate_rack_seq[0, 0]
