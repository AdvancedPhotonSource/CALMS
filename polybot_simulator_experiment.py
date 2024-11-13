# import the following to communicate with the C9 controller
from north import NorthC9  # import the class to communicate with the C9 controller
c9 = NorthC9('A')  # instantiate a C9 controller object with C9 network address A
from Locator import * # import the locations to initiate the robot

# Locator table: specifies the location coordinates of the main modules in the N9 robot that can be called with the goto() function
annealing_block = [-199, 32178, 25032, 19500]
camera_actuator = [-932, 24381, 16346, 16500]

clamp = [2590, 7540, 46677, 23100] # location of the clamp to pick up bernoulli gripper
p_remover = [-507, 26934, 6862, 8000]
sbs_heater = [[-1208, 39893, 14823, 17130], [-1093, 41104, 20127, 17130], [-1071, 38565, 15649, 17130], [-967, 39606, 20332, 17130], [-950, 37052, 15708, 17130], [-859, 37980, 19841, 17130], [-837, 35351, 15189, 17130], [-759, 36214, 18880, 17130], ]
slide_coater = [829, 12700, 37696, 22460]
spect_tower = [-434, 33497, 21706, 16800]

#p_rack_lower and p_rack_upper are the locations of the pipetes
p_rack_lower = [[-1259, 38082, 9957, 14400], [-1255, 38491, 10863, 14400], [-1249, 38890, 11800, 14400], [-1229, 37879, 10318, 14400], [-1224, 38282, 11228, 14400], [-1218, 38675, 12168, 14400], [-1199, 37659, 10632, 14400], [-1194, 38056, 11544, 14400], [-1187, 38443, 12483, 14400], [-1170, 37422, 10901, 14400], [-1164, 37813, 11811, 14400], [-1157, 38194, 12747, 14400], [-1141, 37169, 11125, 14400], [-1135, 37554, 12033, 14400], [-1128, 37928, 12963, 14400], [-1113, 36899, 11306, 14400], [-1107, 37279, 12209, 14400], [-1099, 37647, 13132, 14400], [-1085, 36614, 11446, 14400], [-1079, 36989, 12343, 14400], [-1071, 37351, 13257, 14400], [-1057, 36313, 11546, 14400], [-1051, 36683, 12435, 14400], [-1043, 37040, 13340, 14400], [-1030, 35997, 11606, 14400], [-1024, 36363, 12488, 14400], [-1016, 36715, 13382, 14400], [-1003, 35665, 11629, 14400], [-997, 36027, 12503, 14400], [-989, 36375, 13386, 14400], [-976, 35318, 11616, 14400], [-970, 35677, 12481, 14400], [-963, 36020, 13353, 14400], [-950, 34955, 11566, 14400], [-944, 35311, 12423, 14400], [-937, 35652, 13285, 14400], [-924, 34576, 11482, 14400], [-918, 34930, 12330, 14400], [-911, 35268, 13181, 14400], [-897, 34180, 11363, 14400], [-892, 34533, 12203, 14400], [-885, 34869, 13044, 14400], [-871, 33766, 11209, 14400], [-866, 34119, 12042, 14400], [-859, 34454, 12874, 14400], [-844, 33333, 11020, 14400], [-839, 33687, 11847, 14400], [-833, 34022, 12670, 14400], ]
p_rack_upper = [[-1265, 36442, 6573, 13000], [-1268, 36889, 7395, 13000], [-1269, 37326, 8233, 13000], [-1237, 36260, 6914, 13000], [-1239, 36704, 7745, 13000], [-1240, 37136, 8591, 13000], [-1210, 36060, 7216, 13000], [-1211, 36500, 8054, 13000], [-1211, 36929, 8907, 13000], [-1182, 35843, 7481, 13000], [-1183, 36280, 8325, 13000], [-1182, 36705, 9181, 13000], [-1155, 35609, 7708, 13000], [-1155, 36043, 8556, 13000], [-1154, 36464, 9415, 13000], [-1127, 35358, 7898, 13000], [-1128, 35789, 8748, 13000], [-1126, 36207, 9608, 13000], [-1100, 35089, 8050, 13000], [-1100, 35518, 8902, 13000], [-1099, 35933, 9762, 13000], [-1073, 34804, 8166, 13000], [-1073, 35231, 9019, 13000], [-1071, 35643, 9878, 13000], [-1047, 34502, 8246, 13000], [-1046, 34927, 9098, 13000], [-1044, 35336, 9956, 13000], [-1020, 34183, 8290, 13000], [-1019, 34606, 9142, 13000], [-1017, 35014, 9997, 13000], [-993, 33845, 8298, 13000], [-993, 34269, 9149, 13000], [-991, 34674, 10001, 13000], [-966, 33490, 8270, 13000], [-966, 33913, 9121, 13000], [-964, 34318, 9970, 13000], [-939, 33115, 8206, 13000], [-939, 33539, 9056, 13000], [-937, 33945, 9904, 13000], [-912, 32720, 8106, 13000], [-912, 33147, 8956, 13000], [-911, 33553, 9802, 13000], [-885, 32303, 7968, 13000], [-885, 32733, 8819, 13000], [-884, 33142, 9664, 13000], [-858, 31863, 7792, 13000], [-858, 32298, 8645, 13000], [-857, 32710, 9489, 13000], ]

# rack_48x8ml contains the vials with the solutions
rack_48x8ml = [[-23, 41337, 47608, 22200], [69, 40078, 47475, 22200], [152, 38749, 46930, 22200], [229, 37345, 46100, 22200], [305, 35844, 45042, 22200], [382, 34217, 43765, 22200], [-126, 40736, 43827, 22200], [-18, 39562, 44210, 22200], [75, 38291, 44076, 22200], [162, 36925, 43569, 22200], [246, 35447, 42763, 22200], [330, 33828, 41682, 22200], [-195, 39922, 40473, 22200], [-83, 38839, 41149, 22200], [16, 37636, 41296, 22200], [110, 36314, 41034, 22200], [199, 34863, 40429, 22200], [290, 33248, 39505, 22200], [-235, 38941, 37511, 22200], [-125, 37942, 38315, 22200], [-23, 36802, 38628, 22200], [72, 35524, 38528, 22200], [166, 34092, 38065, 22200], [261, 32471, 37250, 22200], [-253, 37823, 34850, 22200], [-147, 36892, 35680, 22200], [-46, 35803, 36068, 22200], [50, 34557, 36059, 22200], [147, 33132, 35677, 22200], [246, 31482, 34911, 22200], [-252, 36578, 32392, 22200], [-152, 35695, 33188, 22200], [-53, 34641, 33587, 22200], [43, 33407, 33605, 22200], [142, 31964, 33241, 22200], [246, 30241, 32451, 22200], [-237, 35201, 30049, 22200], [-141, 34344, 30773, 22200], [-45, 33301, 31133, 22200], [51, 32051, 31121, 22200], [153, 30544, 30702, 22200], [264, 28658, 29776, 22200], [-209, 33669, 27734, 22200], [-117, 32812, 28360, 22200], [-22, 31745, 28635, 22200], [76, 30427, 28523, 22200], [183, 28764, 27940, 22200], [310, 26475, 26621, 22200], ]


tool_mount = [2650, 29144, 40481, 20814] # the location to mount the bernoulli gripper, once going to this location you nedd to close the gripper to pick up the bernoulli gripper and open the gripper to release the bernoulli gripper

# slide_rack_48x20mm contains the location of the substrates the first list point indicates the lower-left location of the rack
slide_rack_48x20mm = [[156, 11752, 44094, 12800], [156, 11752, 44094, 11200], [156, 11752, 44094, 9600], [156, 11752, 44094, 8000], [156, 11752, 44094, 6400], [156, 11752, 44094, 4800], [156, 11752, 44094, 3200], [156, 11752, 44094, 1600], [55, 12777, 43563, 12800], [55, 12777, 43563, 11200], [55, 12777, 43563, 9600], [55, 12777, 43563, 8000], [55, 12777, 43563, 6400], [55, 12777, 43563, 4800], [55, 12777, 43563, 3200], [55, 12777, 43563, 1600], [-57, 14210, 43561, 12800], [-57, 14210, 43561, 11200], [-57, 14210, 43561, 9600], [-57, 14210, 43561, 8000], [-57, 14210, 43561, 6400], [-57, 14210, 43561, 4800], [-57, 14210, 43561, 3200], [-57, 14210, 43561, 1600], [-191, 16333, 44374, 12800], [-191, 16333, 44374, 11200], [-191, 16333, 44374, 9600], [-191, 16333, 44374, 8000], [-191, 16333, 44374, 6400], [-191, 16333, 44374, 4800], [-191, 16333, 44374, 3200], [-191, 16333, 44374, 1600], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], ]

probe_station = [-1, -1, -1, -1]


# These are the location settings for the rack that contains the solvents and the polymers 
vials_list = [
            # A 6x8 rack, top-right corner is at index [0,0]
            # e.g., 'NaCl' is at index [0,6] of the rack
            # 'None' means empty, 'False' means do not use (keep empty).
            ['water_gap', 'NaCl', None, None, None, None, None, 'polymer_A'],
            [False, None, None, None, None, None, None, None],
            [False, None, None, 'carbon_black', None, None, None, None],
            [None, None, False, None, None, None, None, None],
            [None, None, False, None, None, None, None, None],
            [None, None, None, None, None, None, None, None],
        ]


# Move commands broken down by units taken as argument:

# Move in units of encoder counts: 0 encoder counts on each axis corresponds to the
# home position of that axis. Encoder counts are always integers. Encoder counts are
# never negative, since that would crash the robot into itself past the home position.
# The exception to this is the gripper axis, which has unlimited movement in CW and
# CCW directions. The angle travlled depends on the gearing of each joint:

# gripper (axis 0): each encoder count is 1/4000th of a revolution
# elbow (axis 1): each encoder count is 1/51000th of a revolution
# shoulder (axis 2): each encoder count is 1/101000th of revolution
# z_axis (axis 3): each encoder count is 1/100th of a mm

c9.move_axis(c9.GRIPPER, 1000) # 1/4 turn of the gripper from the home position
c9.move_axis(c9.ELBOW, 10000) # 2*pi*(10000/51000) = 1.23 rad from home position
c9.move_axis(c9.SHOULDER, 20000) # 2*pi*(20000/101000) = 1.24 rad from home position
c9.move_axis(c9.Z_AXIS, 10000) # 100mm from home position


# move the two specified axes to the specified location in counts.
# this command is depricated and will be removed in a future release.
c9.move_sync(c9.ELBOW, c9.SHOULDER, 20000, 30000)

# move all four robot axes to the specified positions in counts:
c9.move_robot_cts(-2000, 10000, 20000, 5000)

# Move in radians:
from math import pi

# For rotational axes (not z_axis), moves may be commanded in units of radians.
# The robot's configuration in radians is with respect to its '0' configuration,
# not its home configuration, which can be found by moving each roational axis to
# 0 radians as follows:
c9.move_axis_rad(c9.GRIPPER, 0)
c9.move_axis_rad(c9.ELBOW, 0)
c9.move_axis_rad(c9.SHOULDER, 0)

c9.move_axis_rad(c9.GRIPPER, pi)
c9.move_axis_rad(c9.ELBOW, pi/2)
c9.move_axis_rad(c9.SHOULDER, -pi/2)

# The move_robot(...) command takes angles in radians for the first 3 joints and a
# height in mm for the z-axis(the height of the bottom of the grippers from the deck)
c9.move_robot(-pi, -pi/2, pi/2, 200)

# Conversion from counts to radians and the reverse is possible:
counts_value = 20000
rad_value = c9.counts_to_rad(c9.SHOULDER, counts_value)
rad_value = -pi/4
counts_value = c9.rad_to_counts(c9.ELBOW, rad_value)


# Moving in task space (mm):
# These moves place the center of the gripper at the specified x, y, z location
# with respect to the origin - on the surface of the deck directly below the axis
# of rotation of the shoulder joint. +Z points up, +Y points away from the column,
# and, following the right-hand-rule, +X points left when looking at the robot from
# the front.

c9.move_axis_mm(c9.Z_AXIS, 150)  # moves so that the bottom of the grippers are 150mm above the deck
c9.move_xy (-150, 0)  # move the elbow and shoulder so that the gripper is at the given xy
c9.move_xyz (-225, 18.75, 288) # move elbow, shoulder, and z_axis to the given task space coordinate

# Examples of passing the shoulder_preference parameter to change the inverse-kinematic solution.
# May also be passed to move_xy:
c9.move_xyz (100, 200, 200, shoulder_preference=c9.SHOULDER_OUT)
c9.move_xyz (100, 200, 200, shoulder_preference=c9.SHOULDER_CENTER)

# To read the current position (in counts) of a given axis:
axis_position = c9.get_axis_position(c9.GRIPPER)

# To read the current positions (in counts) of all robot axes (0-3):
list_of_positions = c9.get_robot_positions()

# Moving to a configuration from the Locator table:
# For each point in the locator table, a list object specifying the robot configuration
# in encoder counts is created. For the test_point object in this project, the Locator.py
# file contains:

# test_point = [0, 34306, 37615, 0]

# The goto() method accepts a location from the Locator table and moves the robot there.
c9.goto(clamp)

# Often, moving from one location to another requires avoiding other modules by first
# moving up to the maximum height, then moving the tool the desired (x,y) coordinate,
# then descending to the appropriate height. All of this can be achieved in one command
# using:
# c9.goto_safe(clamp_dropoff)

# Instruction to pick up a vial and move it to the clamp to uncap, assuming that he gripper has alredy pick up the vial from the correct location
# on the vial rack
c9.close_gripper()  # commend to close the gripper to pick up the vial from the vial rack
c9.open_clamp() # opens the clamp to let the vial get in and out the clamp
c9.open_gripper()  # commend to open the gripper to release the vial once arriving in the clamp location
c9.close_clamp() # closes the clamp to hold the vial and let the gripper to uncap it using the corect uncupping funtions showb below
c9.open_clamp() # opens the clamp to release the vial so that the gripper can pick it up.


# To uncap a vial, you must:
#     - know its pitch (mm/rev)
#     - know its # of revs to unscrew (+ 0.5 for safety) from a typically tight position
#     - be grasping the vial at the appropriate location on the cap (several mm of contact height
#       on the cap without grasping the vial itself)
#     - the vial must be held in the clamp
#     - optionally specifiy a rotational velocity (cts/sec) and acceleration (cts/sec^2)
#     - call c9.uncap(pitch, revs, vel, accel)

# For example, assuming all components were properly positioned, the following command would move
# the gripper and z-axis in coordination so as to uncap a vial with 1.5mm thread pitch, requiring
# 2 revs to fully disengage the threads, at 10000 cts/sec accelerating at 50000 cts/sec^2:
c9.uncap(1.5, 2, 10000, 50000)

c9.move_z(288)  #move out of the way

#return to the vial to recap it
c9.move_z(125 + 3)  #1.5mm thread pitch * 2 revs = 3mm above the height clamped height

# To cap a vial, you must:
#     - know its pitch (mm/rev)
#     - know its # of revs to pre-screw from a height where the threads are disengaged. Pre-screw
#       revolutions do not check the capping torque but are much faster. Typically, 0.5 revs fewer
#       than was required to uncap the vial should be specified here if the capping operation is
#       starting from the same z-height at which the uncapping operation ended. If the uncapping
#       operation performed 2.5 revolutions and finished at a z-height of 100mm, a capping
#       operation beginning at the same height should pre-screw for 2 revs. This leaves a final
#       0.5 revs to tighten to the specified torque.
#     - specify a torque threshold in mA for cap tightness. Typical values are 1000-2000mA
#     - be grasping the vial at the appropriate location on the cap (several mm of contact height
#       on the cap without grasping the vial itself)
#     - the vial must be held in the clamp
#     - optionally specifiy a rotational velocity (cts/sec) and acceleration (cts/sec^2)
#     - call c9.cap(pitch, pre-screw, torque, vel, accel)

# For example, assuming all components were properly positioned, the following command would move
# the gripper and z-axis in coordination so as to cap a vial with 3mm thread pitch with 1.5
# revs of pre-screw, at 5000 cts/sec accelerating at 20000 cts/sec^2 until a motor current
# (proportional to torque) of 1750mA is detected.  
c9.cap(1.5, 2, 1750, vel=5000, accel=20000)

# replace the vial:
c9.open_clamp()
c9.open_gripper()
c9.move_z(288)


# Velcoity and acceleration:

# The velocity and acceleration for any of the move commands can be specified with the
# 'vel' and 'accel' keyword arguments. Units of velocity are always encoder counts/second
# and units of acceleration are always encoder counts/sec^2:
c9.move_axis_rad(c9.SHOULDER, 0, vel=50000, accel=300000)

# For multi-axis movements, the given velocity and acceleration will be for the axis
# travelling the farthest (in terms of counts), and the other axes will have their
# movement vel and accel reduced proportionally so that all movements are synchronous:
c9.move_robot(0, 0, 0, 200,vel=20000, accel=200000)

# Maximum velocity is 100000, maximum acceleration is 500000. For best results, acceleration
# should be ~10x velocity. Too low of an acceleration means the axis will not reach top speed,
# and too high decreases smoothness of motion and ultimately servo performance. A typical
# experiment speed might be 50k velcoity, 300k acceleration.

# If velocity and acceleration are not provided in a command, a default value will be assumed.
# This is initially set at v=5000, a=50000. They may be changed using:
c9.default_vel = 30000
c9.defualt_accel = 150000
c9.move_xyz(-84, 225.25, 212)


# To set True to activate the vaccum on the bernoulli gripper or False to deactivate:
c9.set_output(4, True) # activate the vaccum on the bernoulli gripper to pick up substrates
c9.set_output(4, False)  # deactivate the vaccum bernoulli gripper to releaze substrates

# home a specific axis
c9.home_axis(c9.ELBOW)

# home all four robot axes
c9.home_robot()

#PUMPS:
# the controller supports up to 15 pumps addressed 0-14
# for this example let's say the pump at addr 0 pumps hydrogen peroxide, so we'll name it for
# convenience
H2O2_PUMP = 0

# always home the pump at the beginning of the script
c9.home_pump(H2O2_PUMP)

# set the valve to the left connection:
c9.set_pump_valve(H2O2_PUMP, NorthC9.PUMP_VALVE_LEFT)

# currently the pump accepts movement commands (to aspirate or dispense) as a fraction of
# syringe travel over 3000
# that is, the smallest fluid movement is 1/3000 the syringe's volume
# there is a coarser setting for 1/24000th steps which allows more precision (but slower)
# if speed is not a concern

# aspirate 1/3rd of syringe volume
c9.move_pump(H2O2_PUMP, 1000)


# aspirate/dispense speed may be set to one of 41 presets, numbered 0 - 40. 0 is the fastest,
# 40 is the slowest. I can provide a manual describing exact speeds. Flow rates are,
# of course, determined by the volume of the syringe.

# set pump speed to fastest
c9.set_pump_speed(H2O2_PUMP, 0)

# set pump to the right connection:
c9.set_pump_valve(H2O2_PUMP, NorthC9.PUMP_VALVE_RIGHT)

# dispense 1/6th of syringe volume (pump was at position 1000, move to position 500,
# 500/3000 delta = 1/6)
c9.move_pump(H2O2_PUMP, 500)

# move the valve to the center connection to flush the lines.
c9.set_pump_valve(H2O2_PUMP, NorthC9.PUMP_VALVE_CENTER)

# If using the NorthIDE to program the robot, volumes of the syringes in each pump may be
# specified. In this project the pump at addr 0 is set to a volume of 1mL.
# This enables the following volumetric commands:

#aspirate 0.5mL:
c9.aspirate_ml(H2O2_PUMP, 0.5)
#dispense 0.2mL:
c9.dispense_ml(H2O2_PUMP, 0.2)


