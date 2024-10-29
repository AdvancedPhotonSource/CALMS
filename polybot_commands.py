#north_c9 API reference v0.3

from north import NorthC9  # import the class to communicate with the C9 controller
from Locator import *  # import the contents of the Locator table (View -> Locator)

# contains the constants used to configure the analog chip, no need to import if you are
# not calling c9.config_analog(...)
from north import ADS1115  

c9 = NorthC9('A')  # instantiate a C9 controller object with C9 network address A

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

# Note that an increase in counts value for the ELBOW joint corresponds to a CW rotation,
# whereas an increasing counts value represents a CCW rotation for the SHOULDER.

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
print(f"A position of {counts_value} counts maps to an angle of {rad_value:.3f} for the SHOULDER")
rad_value = -pi/4
counts_value = c9.rad_to_counts(c9.ELBOW, rad_value)
print(f"An angle of {rad_value:.3f} maps to a position of {counts_value} for the ELBOW")


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
c9.goto(clamp_dropoff)

# Elements in a grid of locations can also be referenced:
c9.goto(vial_rack[0])  #note the collision that happens here fixed with goto_safe, below

# Often, moving from one location to another requires avoiding other modules by first
# moving up to the maximum height, then moving the tool the desired (x,y) coordinate,
# then descending to the appropriate height. All of this can be achieved in one command
# using:
c9.goto_safe(clamp_dropoff)

# To safely move the tool to an (x,y) coordiate of a location without descending, use:
c9.goto_xy_safe(vial_rack[0])

# To move to the z-coordinate only of a location:
c9.goto_z(vial_rack[0])


# Now we can pick up a vial and move it to the clamp to uncap:
c9.close_gripper()
c9.goto_safe(clamp_dropoff)
c9.close_clamp()

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
c9.goto_safe(vial_rack[0])
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

# Other commands: These are self-explanatory. There is not vel/accel for these as they
# are controlled pneumatically. They block for ~200ms to allow time to actuate.
c9.close_gripper()
c9.close_clamp()
c9.open_gripper()
c9.open_clamp()
c9.bernoulli_on() #activate the bernoulli gripper
c9.bernoulli_off() #deactivate the bernoulli gripper

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


### UNSIMULATED COMMANDS ###
# These commands will have no effect in the simulator, though they will be helpful
# for operating the robot. Some of these commands may be present in the "Sim Inputs"
# tab, allowing the user to generate dummy measurements in the simulator useful for
# testing workflow and data pipelines.

#The get_info() command is used to fetch the FW version and is a useful command for
#checking connectivity with a controller
c9.get_info()

# To enable (True) or disable (False) the servo action on a specific axis, use this command:
c9.axis_servo(c9.Z_AXIS, False)

# To enable/disable the servo action on robot axes (0-3) use this command:
c9.robot_servo(True)

# The C9 controller has 32 (0-31) digital outputs, some of which are dedicated to things like the
# gripper or weigh scale clamp.
# To set (True) or clear (False) a digital output, in this case set output number 13 high:
c9.set_output(13, True)

# The C9 controller has 32 (0-31) digital inputs, some of which are dedicated to things like the
# e-stop and motor fault monitoring.
# To read the state of a digital input, in this case input number 22:
digital_value = c9.get_input(22)

# The C9 controller has 2 (0-1) ADS1115 analog to digital converters, see datasheet here:
# http://www.ti.com/lit/ds/symlink/ads1115.pdf
# By default, both are configured to be continuously reading the voltage between AIN0 and GND
# with a range of 4.096V at 64 samples/sec
# To configure the chip in another way:

# This configures ADC 0 to start a differential reading betweens inputs 0 and 1 with a
# full-scale range of 0.256V, updating its value continuously at 8 samples/sec:
c9.config_analog(0,
                 ADS1115.START_READ,
                 ADS1115.AIN0_AIN1,
                 ADS1115.V0_256,
                 ADS1115.CONTINUOUS,
                 ADS1115.SPS8
                 )

# Note that the samples per second parameter specifies how often the ADC will update its internal
# value. The communication between the controller and the PC limits the effective update rate to
# somewhere between 32 and 64 samples/sec.

# To read the analog input (in this case ADC 1):
analog_value = c9.get_analog(1)

# To zero the weigh scale:
c9.zero_scale()

# To read the current value from the weigh scale:
is_steady, weight = c9.read_scale()

# To wait for a steady scale reading:
steady_weight = c9.read_steady_scale()             