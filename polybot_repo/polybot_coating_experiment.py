import loca  # location information
import pandas as pd
import robotics as ro
from robotics import procedure as proc
from robotics.workflow.fields import (  # noqa: F401
    Name,
    Number,
    Speed,
    Temperature,
    Time,
    Volume,
)

# fmt: off
ro.runtime['rack_status'] = {
    'vial': pd.DataFrame(
        [
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
        [
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
# fmt: on


# --------------------------------------------------------------
# hardware modules
c9 = ro.system.init('controller')  # N9 robot controller
t8 = ro.system.init('temperature')  # temperature controller
coater = ro.system.init('coater')  # coating station
camera = ro.system.init('camera')  # camera at the coating stationo

# The N9 robot arm has one linear motor 'z' for vertial movement
# along the z-axis. It also has two rotational motors
# 'shoulder' and 'elbow', and together they control the x- and y-
# positioning of the arm.


# --------------------------------------------------------------
# initialize the experimemntal workflow
wf = ro.workflow.init(__file__, name="Film Coating Experiment")


# --------------------------------------------------------------
# Functions decorated by @ro.workflow.register are part of the
# experimental workflow. They are executed in order.
# smp means 'sample', it is a Python dictionary-like object.
# smp must be passed as the first argument in every workflow function.


@ro.workflow.register(workflow=wf)
def init_system(smp):
    """Initialize the system"""

    ro.reload(loca)
    ro.system.reset()

    coater.clean_blade(clean_time=3, scratch_time=10)


@ro.workflow.register(workflow=wf)
def prepare_substrate(smp):
    """place a new substrate on the coating station"""

    # prepare up a new substrate for coating
    c9.tool = 'substrate_tool'  # pick up the bernoulli substrate gripper tool
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
    c9.set_output('coater_stage_vacuum', True)
    c9.set_output('substrate_tool', False)
    c9.tool = None  # drop off the tool in the gripper

    # take a picture of the substrate
    img_name, img = camera.take_image(
        'substrate.jpg',
        focus=35,
        crop=[0, 900, 600, 1550],
        name_only=False,
    )
    smp['raw_outputs'][camera.outkey][
        'substrate'
    ] = img  # add image to the sample dictionary
    smp.save()  # save data to file


@ro.workflow.register(workflow=wf)
def coating_on_top(
    smp,
    # fmt: off
    sol_label: str = Name(
        'Name of the coating solution',
        choices=[
            'NaCl',
            'carbon_black',
            'polymer_A',
        ],
    ),
    V: float = Volume(
        'Volume of coating solution (mL)',
        constant=True, default=0.003,
    ),
    vel: float = Speed(
        'Coating velocity (mm/sec)',
        ge=0.5, le=2, step=0.1, default=1,
        # ge means great than or equal
        # le means less than or equal
        # step is the increment
    ),
    T: float = Temperature(
        'Coating temperature (Celcius)',
        ge=50, le=180, step=10, default=50,
    ),
    # fmt: on
):
    """Film coating"""

    # set the coating temperature
    t8.set_temp(1, T)

    # move solution from the vial rack to the clamp
    vial_index = proc.find_rack_index('vial', sol_label)
    c9.position = loca.vial_rack[vial_index]  # move robot arm to the solution
    c9.set_output('gripper', True)  # close the robot arm gripper
    c9.position = loca.clamp  # move robot arm to the clamp
    c9.set_output('clamp', True)  # close the clamp
    c9.set_output('gripper', False)  # open the robot arm gripper

    # aspirate the solution in the clamp
    proc.new_pipette(c9)  # get a new pipette
    c9.position = loca.clamp
    c9.set_output('gripper', True)
    c9.uncap(pitch=1.75, revs=3.0, vel=5000, accel=5000)  # uncap the vial
    uncap_position = c9.position  # record the position
    c9.position = loca.p_clamp  # move pipette to the clamp, inside the vial
    c9.aspirate_ml(0, V)  # first argument is the pump ID
    c9.move_axis(
        'z', c9.position[3] - 9000, vel=15000
    )  # quickly move up by 9 cm
    c9.position = uncap_position  # move gripper back to the recorded position
    c9.cap(pitch=1.75, revs=3.0, torque_thresh=1000, vel=5000, accel=5000)
    c9.set_output('gripper', False)
    c9.move_axis(
        'z', c9.position[3] - 5000, vel=15000
    )  # quickly move up by 5 cm

    # dispense solution
    coater.position = 45  # move coater blade to the starting position
    c9.position = loca.p_coater  # move pipette to the coating station
    c9.dispense_ml(0, V)  # first argument is the pump ID
    c9.move_axis('z', 0)  # move robot arm all the way up
    coater.velocity = vel  # set the coating velocity
    coater.position = 75  # move blade all the way to the right

    proc.remove_pipette(c9)

    # record image data of the film
    img_name, img = camera.take_image(
        'coated_film.jpg',
        focus=35,
        crop=[0, 900, 600, 1550],
        name_only=False,
    )
    smp['raw_outputs'][camera.outkey][
        'coated_film'
    ] = img  # add image to the sample dictionary
    smp.save()  # save data to file

    # return solution in clamp back to the vial rack
    c9.position = loca.clamp  # move robot arm to the clamp
    c9.set_output('gripper', True)  # close the robot arm gripper
    c9.set_output('clamp', False)  # close the clamp
    c9.position = loca.vial_rack[vial_index]  # move robot arm to the solution
    c9.set_output('gripper', False)  # open the robot arm gripper


@ro.workflow.register(workflow=wf)
def store_sample(smp):
    """Return sample to rack, end of experiment"""

    c9.tool = 'substrate_tool'
    c9.position = loca.s_coater
    c9.set_output('substrate_tool', True)
    c9.set_output('coater_stage_vacuum', False)
    index = proc.find_rack_index('substrate', 'last_used')
    c9.position = loca.substrate_rack_seq[index]
    c9.set_output('substrate_tool', False)
    ro.runtime['rack_status']['substrate'].iloc[index] = 'used'
    c9.tool = None
