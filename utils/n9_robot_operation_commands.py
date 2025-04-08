# These are the functions that should always be imported before running the N9 robotic system
import loca # location information
import pandas as pd
import robotics as ro
from robotics import procedure as proc
from lab_setup import rack_status # always import the rack_status module before initializing the system

# Access the rack status to find the location of the vial
vial_rack = rack_status['vial']

# hardware modules
# example: hardware = ro.system.init('hardware_module')
# available hardware modules: controller, temperature, coater
c9 = ro.system.init('controller')  # N9 robot controller
c9.set_temp(1, T)  # temperature controller, where T is coating stage temperature
coater = ro.system.init('coater')  # coating station

c9.tool = 'substrate_tool' # pick up the bernoulli substrate gripper tool, use this when you need to pick up a substrate
c9.tool = None  # drop off the tool in the gripper

# Vaccum functions
c9.set_output('substrate_tool', True) # when True activate the vaccum on the bernoulli gripper to pick up substrates
c9.set_output('substrate_tool', False)  #when False deactivate the vaccum bernoulli gripper to releaze substrates
c9.set_output('coater_stage_vacuum', True) # when True activates the vaccum on the coater stage to hold the substate on the stage, set to false when we need to pick up and move the substrate

# Gripper functions
c9.set_output('gripper', True)  # commend to close the gripper to pick up the vial from the vial rack
c9.set_output('gripper', False)  # commend to open the gripper to release the vial

# Clamp functions
c9.set_output('clamp', False)  # opens the clamp to let the vial get in and out the clamp
c9.set_output('clamp', True)  # closes the clamp to hold the vial. first we close the clamp and then open the gripper.
c9.position = loca.clamp  # move robot arm to the clamp
c9.position = [0, 0, 0, 0] # move robot arm to the initial location

# Film coating
T = None
sol_label = None

# move solution from the vial rack to the clamp
vial_index = proc.find_rack_index('vial', sol_label)
c9.position = loca.vial_rack[vial_index]  # move robot arm to the solution

# capping/uncapping a vial:
c9.cap(pitch=1.75, revs=3.0, torque_thresh=1000, vel=5000, accel=5000) # cap the vial
uncap_position = c9.uncap(pitch=1.75, revs=3.0, vel=5000, accel=5000)  # uncap the vial and record the position
c9.position = uncap_position  # move gripper back to the recorded position

# aspirate the solution in the clamp
c9.aspirate_ml(0, 0.5)  # aspirate 0.5mL
c9.dispense_ml(0, 0.5)  # dispense 0.2mL

# Pick up a pipette
proc.new_pipette(c9)  # get a new pipette
proc.remove_pipette(c9)  # remove the pipette

# Arm available movements:
c9.move_axis('z', 0)  # move robot arm all the way up
c9.move_axis('z', c9.position[3] - 9000, vel=15000)  # quickly move up by 9 cm relative to the current z
c9.move_axis('z', c9.position[3] - 5000, vel=15000)  # quickly move up by 5 cm relative to the current z

c9.position = loca.clamp  # move gripper to the clamp
c9.position = loca.p_clamp  # move pipette to the clamp, inside the vial
c9.position = loca.pipette_coater_one  # move pipette to the coating station

# return solution in clamp back to the vial rack
c9.position = loca.clamp  # move robot arm to the clamp
c9.position = loca.vial_rack[vial_index]  # move robot arm to the solution

# Return sample to rack, end of experiment
c9.position = loca.s_coater # move substrate to the coating station
c9.position = loca.substrate_rack_seq[0, 0]  # move substrate to the substrate rack (top-left position)

# Coater related actions:
coater.position = 45  # move coater blade to the starting position
coater.velocity = 1  # set the coating velocity
coater.position = 75  # move blade all the way to the right
