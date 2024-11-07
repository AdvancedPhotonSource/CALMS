import sys

import numpy as np

import robotics as ro
from robotics import procedure as proc

loca = ro.runtime['loca'] = sys.modules[__name__]
lookup = proc.loc_lookup(loca)

alias = {
    'substrate_tool': 'bernoulli',
    'substrate_vacuum': 'bernoulli',
    'substrate_vacuum_gripper': 'bernoulli',
    'coater_vacuum': 'coater_stage_vacuum',
}

pump = {
    0: 0,
    1: 1,
}

air_output = {
    'gripper': 0,
    'bernoulli': 3,
    'blade_vacuum': 4,
    'clamp': 5,
    'coater_stage_vacuum': 6,
    'camera_stage_a': 7,
    'camera_stage_b': 8,
    'cooking_module_lock': 9,
    'waste_bottle_vacuum': 10,
    'air_extention_12-17': 11,
    'cleaner_air': 12,
    'pin_outer': 13,
    'pin_inner': 14,
    'KLA_vacuum': 15,
    'characterization_clamp': 16,
    'empty': 17,
}

# 5.8 mm pipette, pipette tip to arm 4.4 mm 
# 19.2 cm for 20000, 9.2 cm for 10000, 4.2 cm for 5000
z_offset = {
    'pipette': 0,  # 0 for long, 200 for short
    'substrate': 0,  # 0 for silicon dioxide, -50 for quartz, -100 for ITO
    'bernoulli': 2072,
}

# fmt: off
use_z_offset = True

waypoint_0 = [-262, 6687, 6622, 0]
waypoint_1 = [-2000, 0, 31813, 0]
waypoint_2 = [-2000, 37895, 31813, 0]

bernoulli = [-700, 13293, 24797, 17390 + z_offset['bernoulli']*use_z_offset]
clamp = [-261, 7566, 46711, 16161]

camera_actuator = [-567, 17594, 13329, 20132]
camera_actuator_film = [-310, 11529, 6843, 18560]  # approach from +y
camera_actuator_film_rotate = [944, 10376, 11435, 18560]  # approach from +x

annealing_block_standby = [-275, 35345, 29560, 17670]  # [-203, 32136, 24999, 17670] TODO
_annealing_block_approach = [-194, 32108, 24962, 17670] # [-275, 35420, 29603, 17670] TODO
_annealing_block = _annealing_block_approach[:3] + [18700] #[18770]
annealing_block_seq = proc.SequenceArray(
    annealing_block_standby, _annealing_block_approach, _annealing_block
)

# spect_tower_standby = [483, 31477, 23394, 18594]
spect_tower_standby = [162, 32999, 26282, 18186]
_spect_tower_posX_approach = [-428, 33401, 21676, 18186]
_spect_tower_posX = [-429, 33401, 21676, 18761] # + 40]
_spect_tower_negY_approach = [1029, 24963, 16518, 18186]
_spect_tower_negY = [1029, 24963, 16518, 18761]
spect_tower_seq = proc.SequenceArray(
    spect_tower_standby, _spect_tower_posX_approach, _spect_tower_posX
)

probe_standby = [691, 26444, 36173, 20000]
_probe_approach = [683, 26022, 35078, 20000]
_probe = _probe_approach[:3] + [20650] #[20786]
probe_seq = proc.SequenceArray(probe_standby, _probe_approach, _probe)
_probe_approach_PDMS = [680, 26018, 35005, 20000]
_probe_PDMS = _probe_approach_PDMS[:3] + [20550]
probe_PDMS_seq = proc.SequenceArray(probe_standby, _probe_approach_PDMS, _probe_PDMS)

substrate = [97, 13176, 45442, 1415]
substrate_safe = [181, 10423, 42083, 1415]
s_coater = [825, 12899, 37774, 22300] # use z=20700 to align

# p_clamp = [1557, 5811, 42416, 18400 + z_offset['pipette']]  # original position
p_clamp_20ul = [1557, 5723, 42320, 19150 + z_offset['pipette']]  # side of the vial
p_clamp = [1557, 5811, 42416, 18450 + z_offset['pipette']]  # side of the vial
# p_clamp =[1557, 5578, 42160, 18500 + z_offset['pipette']]  # center of the vial

# p_coater = [70, 13460, 34960, 20456 + z_offset['pipette'] + z_offset['substrate']]
# pipette_coater_20uL = [68, 13393, 34836, 20800 + z_offset['pipette'] + z_offset['substrate']]
pipette_coater_20uL = [68, 13437, 34866, 20800 + z_offset['pipette'] + z_offset['substrate']]
pipette_coater_one = [68, 13393, 34836, 20100 + z_offset['pipette'] + z_offset['substrate']]
p_camera_actuator = [35, 16299, 14389, 18376 + z_offset['pipette']]

# p_PDMS = [78, 28529, 35367, 17794]  # original position
p_PDMS = [78, 28566, 35453, 17794]

pipette_removal_approach = [29, 24600, 6792, 12416]
_pipette_removal = [231, 24448, 6527, 11621]
pipette_removal_seq = proc.SequenceArray(
    pipette_removal_approach, _pipette_removal
)


def pipette_rack_locator(ref=None, ref_index=(0, 0)):
    return proc.rack_locator(
        nrow=2,
        ncol=24,
        row_spacing=9,
        col_spacing=9,
        ref=ref,
        ref_index=ref_index,
        pipette_tip=True,
    )


p_rack_upper = pipette_rack_locator(ref=[1420, 26096, 3286, 10785 + z_offset['pipette']])
# p_rack_lower = pipette_rack_locator(ref=[1428, 28116, 6600, 13224 + z_offset['pipette']])
p_rack_lower = pipette_rack_locator(ref=[1428, 28116, 6600, 13424 + z_offset['pipette']])
pipette_rack = np.concatenate((p_rack_lower, p_rack_upper), axis=0)
pipette_coater = proc.rack_locator(
    nrow=4,
    ncol=1,
    row_spacing=5,
    col_spacing=0,
    ref=[81, 12897, 34509, 20040 + z_offset['pipette']],
    # ref=[81, 12897, 34509, 20000],  # OECT
    ref_index=(0, 0),
    pipette_tip=True,
)


def cooking_rack_locator(ref=None, ref_index=(0, 0), pipette_tip=False):
    return proc.rack_locator(
        nrow=1,
        ncol=4,
        row_spacing=1,
        col_spacing=32,
        ref=ref,
        ref_index=ref_index,
        pipette_tip=pipette_tip,
    )


cooking_rack = cooking_rack_locator(
    ref=[537, 36221, 18886, 16322],
    ref_index=(0, 0),
)


cooking_rack_cap = cooking_rack_locator(
    ref=[-9465, 35067, 14213, 15833 + 300],
    ref_index=(0, 0),
)

# cooking_rack_pipette_approach = cooking_rack_locator(
#     ref=[238, 38082, 28161, 12414],
#     pipette_tip=True,
# )
cooking_rack_pipette = cooking_rack_locator(
    ref=[87, 37757, 23691, 16200],  # old z = 12414
    pipette_tip=True,
)
# cooking_rack_pipette_seq = proc.SequenceArray(
#     cooking_rack_pipette_approach, cooking_rack_pipette
# )


def substrate_rack_locator(ref=None, ref_index=(0, 0)):
    return proc.rack_locator(
        nrow=12,
        ncol=6,
        row_spacing=16,
        col_spacing=30,
        ref=ref,
        ref_index=ref_index,
        tool=70,
        tool_o=np.pi,
        pipette_tip=False,
        substrate_rack=True,
    )


substrate_rack_standby = substrate_rack_locator(ref=[348, 8566, 42651, 1500])
_substrate_rack_approach = substrate_rack_locator(ref=[273, 10882, 45309, 1500])
_substrate_rack = substrate_rack_locator(ref=[273, 10882, 45309, 2050])
substrate_rack_seq = proc.SequenceArray(substrate_rack_standby, _substrate_rack_approach, _substrate_rack)
# for PDMS
_substrate_rack_approach_PDMS = substrate_rack_locator(ref=[273, 10961, 45399, 1470])
_substrate_rack_PDMS = substrate_rack_locator(ref=[273, 10961, 45399, 1920])
substrate_rack_PDMS_seq = proc.SequenceArray(substrate_rack_standby, _substrate_rack_approach_PDMS, _substrate_rack_PDMS)
# for PDMS back
substrate_rack_PDMS_back_seq = proc.SequenceArray(
    substrate_rack_locator(ref=[348, 8566, 42651, 1200]),
    substrate_rack_locator(ref=[328, 9406, 43585, 1200]),
    substrate_rack_locator(ref=[328, 9406, 43585, 1470]),
    substrate_rack_locator(ref=[273, 10961, 45399, 1500]),
)


def vial_rack_locator(ref=None, ref_index=(0, 0)):
    return proc.rack_locator(
        nrow=6,
        ncol=8,
        row_spacing=26,
        col_spacing=26,
        ref=ref,
        ref_index=ref_index,
    )


vial_rack_left = vial_rack_locator(ref=[1159, 34239, 43859, 22021 + 300], ref_index=(5, 7))
vial_rack_right = vial_rack_locator(ref=[235, 857, 24407, 22021 + 300], ref_index=(2, 1))
vial_rack = np.concatenate((vial_rack_left, vial_rack_right), axis=0)
