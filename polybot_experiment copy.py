# These are the functions to be imported before running the N9 robotic system
import loca  # location information
import pandas as pd
import robotics as ro
from robotics import procedure as proc

# These are the location settings for the rack that contains the existing polymers 
# If a polymer does not exist in this list, then respond "Experiment cannot be initiated."
ro.runtime['rack_status'] = {
    'vial': pd.DataFrame(
        [   # These are the location settings for the rack that contains the solvents and the polymers 
            # A 6x8 rack, top-left corner is at index [0,0]
            # e.g., 'NaCl' is at index [0,1] of the rack
            # 'None' means empty, 'False' means do not use (keep empty).
            ['water_gap', 'NaCl', None, None, None, None, None, None],
            [False, None, 'polymer_A', None, None, None, None, None],
            [False, None, None, 'carbon_black', None, None, None, None],
            [None, None, False, None, None, None, None, None],
            [None, None, False, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
        ]
    ),
    'substrate': pd.DataFrame(
        [   # These are the location settings for the rack that contains the substrates
            # A 12x6 rack, 'new' means new, unused substrate
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
            ['new', 'new', 'new', 'new', 'new', 'new'],
        ]
    ),
}

# --------------------------------------------------------------
# hardware modules
# example: hardware = ro.system.init('hardware_module')
# available hardware modules: controller, 
c9 = ro.system.init('controller')  # N9 robot controller
t8 = ro.system.init('temperature')  # temperature controller
coater = ro.system.init('coater')  # coating station



c9.tool = 'substrate_tool' # pick up the bernoulli substrate gripper tool, use this when you need to pick up a substrate
c9.tool = None  # drop off the tool in the gripper

# To set True to activate the vaccum on the bernoulli gripper or False to deactivate:
c9.set_output('substrate_tool', True) # activate the vaccum on the bernoulli gripper to pick up substrates
c9.set_output('substrate_tool', False)  # deactivate the vaccum bernoulli gripper to releaze substrates


# Instruction to pick up a vial and move it to the clamp to uncap, assuming that he gripper has alredy pick up the vial from the correct location
# on the vial rack
c9.set_output('gripper', True)  # commend to close the gripper to pick up the vial from the vial rack
c9.set_output('clamp', True)  # opens the clamp to let the vial get in and out the clamp
c9.set_output('gripper', False)  # commend to open the gripper to release the vial once arriving in the clamp location
c9.set_output('clamp', False)  # closes the clamp to hold the vial and let the gripper to uncap it using the corect uncupping funtions showb below
c9.position = loca.clamp  # move robot arm to the clamp


c9.set_output('coater_stage_vacuum', True) # when True activates the vaccum on the coater stage to hold the substate, set to false when we need to pick up and move the substrate

# prepare up a new substrate for coating
index = proc.find_rack_index(
    'substrate', 'new'
)  # the next available substrate
c9.position = {
    'loc': loca.substrate_rack_seq[index],
    'vel': 5000,
    'accel': 5000,
}  # seq means a sequence of movements
c9.set_output(
    'substrate_tool', True
)  # turn on the substrate gripper vacuum
ro.runtime['rack_status']['substrate'].iloc[index] = 'last_used'
c9.position = {
    'loc': loca.s_coater,
    'vel': 5000,
    'accel': 5000,
}  # move substrate to coating station at a lower speed and acceleration




# Film coating"""
T = None
sol_label = None

# set the coating temperature
t8.set_temp(1, T)

# move solution from the vial rack to the clamp
vial_index = proc.find_rack_index('vial', sol_label)
c9.position = loca.vial_rack[vial_index]  # move robot arm to the solution



# aspirate the solution in the clamp
proc.new_pipette(c9)  # get a new pipette
c9.position = loca.clamp
c9.uncap(pitch=1.75, revs=3.0, vel=5000, accel=5000)  # uncap the vial
uncap_position = c9.position  # record the position
c9.position = loca.p_clamp  # move pipette to the clamp, inside the vial
#PUMPS:
# the controller supports up to 15 pumps addressed 0-14
# for this example let's say the pump at addr 0 pumps hydrogen peroxide, so we'll name it for
# convenience
PUMP_NUM = 0
#aspirate 0.5mL:
c9.aspirate_ml(PUMP_NUM, 0.5)
#dispense 0.2mL:
c9.dispense_ml(PUMP_NUM, 0.2)
c9.move_axis('z', c9.position[3] - 9000, vel=15000)  # quickly move up by 9 cm
c9.position = uncap_position  # move gripper back to the recorded position
c9.cap(pitch=1.75, revs=3.0, torque_thresh=1000, vel=5000, accel=5000)
c9.move_axis('z', c9.position[3] - 5000, vel=15000)  # quickly move up by 5 cm

# dispense solution
coater.position = 45  # move coater blade to the starting position
c9.position = loca.p_coater  # move pipette to the coating station
c9.dispense_ml(0, 0.05)  # first argument is the pump ID
c9.move_axis('z', 0)  # move robot arm all the way up
coater.velocity = 1  # set the coating velocity
coater.position = 75  # move blade all the way to the right
proc.remove_pipette(c9)

# return solution in clamp back to the vial rack
c9.position = loca.clamp  # move robot arm to the clamp
c9.position = loca.vial_rack[vial_index]  # move robot arm to the solution


# Return sample to rack, end of experiment
# step ? pick up the 
c9.position = loca.s_coater
index = proc.find_rack_index('substrate', 'last_used')
c9.position = loca.substrate_rack_seq[index]
ro.runtime['rack_status']['substrate'].iloc[index] = 'used'
