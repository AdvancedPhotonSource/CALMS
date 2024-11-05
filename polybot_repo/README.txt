# ----- prepare AI training data -----
# input samples/CD_*.json, output samples/posted/AI_*.json

cd OECT_demo_KL
python post_to_AI.py

# ----- Active learning: AI suggestion -----
# input samples/posted/AI_*.json, output samples/_*.json

cd activelearning
# stop the file observer on robotics if you don't want exp to start
python run_exp_loop.py ml_settings_OECT.yml --run_once


# ----- Stop the robot --------
soft stop:  ctrl+c in the terminal
hard stop:  press the emergency button

# ------- Reset the robot --------

if emergency button is pressed, slightly rotate it to pop it out

turn off c9 and t8 controllers

move robot to the reset position (elbow pointing left, collasped)

turn on c9 before t8

>>> ctrl+z enter to exit python

follow the Manual control section then run
>>> c9.home_robot()

# ------- move robot manually -----
c9.robot_servo(0)

# ------- reset coater -----
coater.reset()      

# ------- Manual control ---------

open Visual Studio Code

View > Terminal  (ctrl+`)

PS C:\Users\Public\robot> cd .\OECT_demo_KL\
PS C:\Users\Public\robot\OECT_demo_KL> python
Python 3.9.9 | packaged by conda-forge | (main, Dec 20 2021, 02:36:06) [MSC v.1929 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license" for more information.
>>> 

import robotics as ro
from robotics import procedure as proc
import loca
import rack_status_OECT
ro.debug_on()
c9 = ro.system.init('controller')
coater = ro.system.init('coater')

c9.home_robot()

Keithley = ro.system.init('IV')
Keithley.reset()



# ---------- Checklist -------------

+ launch KXCI, connect Keithley cables (blue with blue, match 1 2 3)

+ adjust blade, 50 um gap, 1 mm out from the edge

coater.reset()
coater.velocity = 10
c9.set_output('blade_vacuum', True)
c9.set_output('coater_stage_vacuum', True)
coater.position = 0  # install blade
coater.position = 45.5  # coating start position
coater.position = -70  # retracted position
c9.set_output('coater_stage_vacuum', False)

+ set up cleaning swab

+ open camera and check the position

+ turn on the LED light, change to the lowest brightness

+ reload pipettes, substrates, solution vials


# ------ start autonomous run -------

cd .\OECT_demo
python -m robotics   # add --sim for simulated mode

in a web browser, http://127.0.0.1:5000

start the observer

# ------ start AI ------

cd .\activate_learning
python .\example_polybot_ml_loop.py


# ------ procedures -------
ro.system.reset()

# moving vial from rack to heating block
 # places robot arm back to origin
c9.position = loca.vial_rack[0,1] # [0,0] is top left vial
# go to loca.py "air_output" for names of objects in Polybot
c9.set_output('gripper', True)
c9.position = loca.cooking_rack[0,0] # the rack is called "cooking_rack" (in loca.py), the status of this rack is "cooking" (in rack_status_OECT.py)
c9.set_output('gripper', False)

# move substrate from rack to coating station
coater.position = -70 #moving coater position just in case
c9.tool = 'substrate_tool'  #moves robot arm to grab subtrate tool (the piece with vacuum to grab substrates)
c9.position = loca.substrate_rack_seq[1,0] # go to rack location
c9.set_output('substrate_tool', True) # turns on vacuum of substrate tool, "grabs" a substrate
c9.position = loca.s_coater # moves substrate to coater
c9.set_output('coater_stage_vacuum',True) # opens vacuum on coater so substrate sticks to it
 # releases vacuum on substrate holder so it can move away
c9.tool = None # robot arm returns substate holder to original location and leaves it there

coater.position = -70 
c9.tool = 'substrate_tool'  
c9.position = loca.s_coater
c9.set_output('substrate_tool', True) 
c9.set_output('coater_stage_vacuum', False) 
c9.position = loca.substrate_rack_seq[1,0] 
c9.set_output('substrate_tool', False) 
c9.tool = None

t8 = ro.system.init('temperature')
t8.set_temp(0, 60) #cooking rack
t8.set_temp(1, 25) #coater
t8.set_temp(2, 140) #anneal 

# move vial from cooking_rack to clamp 
c9.position = loca.cooking_rack[0,0]
c9.set_output('gripper', True)
c9.position = loca.clamp
c9.set_output('clamp', True)
c9.set_output('gripper', False)

# pipette solution on substrate (similar to function aspirate_clamp_solution, not blade coater is not engaged here)
c9.position = loca.pipette_rack[0,2] #picks up pipette already
c9.position = loca.clamp
c9.set_output('gripper', True)
c9.uncap(pitch=1.75, revs=3.0, vel=5000, accel=5000)
uncap_z_clamp = c9.get_axis_position(3)  # records uncap position
# c9.position = loca.p_clamp_20ul # puts pipette into vial in clamp 18400
# c9.goto_safe([1557, 5578, 42160, 19050], vel=None, accel=None)
c9.goto_safe([1557, 5811, 42416, 19050], vel=None, accel=None)
c9.aspirate_ml(0, 0.005)  # 1st arg is pump number
c9.goto_xy_safe(loca.clamp)

c9.goto_xy_safe(loca.clamp)
c9.move_axis('z', uncap_z_clamp)
c9.cap(pitch=pitch, revs=revs, torque_thresh=1000, vel=5000, accel=5000)
c9.set_output('gripper', False)
c9.position = loca.pipette_coater_one
c9.dispense_ml(0, 0.005) 
c9.position = loca.pipette_coater_one[:3] + [0]#moves the robot arm up and out of the way
c9.position = {'loc': loca.pipette_removal_seq, 'vel': [10000, 1000], 'accel': [10000, 1000], 'need_undo': False} #removes tip into waste
c9.move_axis(3, 0)
c9.position = [0,0,0,0]

# testing volatile solvent
c9.position = loca.cooking_rack[0,0]
c9.set_output('gripper', True)
c9.position = loca.clamp
c9.set_output('clamp', True)
c9.set_output('gripper', False)

c9.position = loca.pipette_rack[0,0] #picks up pipette already
c9.position = loca.clamp
c9.set_output('gripper', True)
c9.uncap(pitch=1.75, revs=2, vel=5000, accel=5000)
uncap_z_clamp = c9.get_axis_position(3)  # records uncap position
c9.goto_safe([1557, 5723, 42320, 19150], vel=None, accel=None)
c9.aspirate_ml(0, 0.005+0.005)  # 1st arg is pump number
c9.dispense_ml(0, 0.005+0.006) 
c9.goto_safe([1557, 5723, 42320, 19150], vel=None, accel=None)
c9.aspirate_ml(0, 0.005+0.002)
c9.goto_xy_safe(loca.clamp)
c9.move_axis('z', uncap_z_clamp)
c9.cap(pitch=1.75, revs=2, torque_thresh=1000, vel=5000, accel=5000)
c9.set_output('gripper', False)
coater.position = 45.5
c9.goto_safe([68, 13393, 34836, 20800], vel=None, accel=None)
c9.dispense_ml(0, 0.005) 
c9.position = loca.pipette_coater_20uL[:3] + [0]#moves the robot arm up and out of the way
c9.position = {'loc': loca.pipette_removal_seq, 'vel': [10000, 1000], 'accel': [10000, 1000], 'need_undo': False} #removes tip into waste
c9.move_axis(3, 0)

c9.position = loca.clamp
c9.set_output('gripper', True)
c9.set_output('clamp', False)
c9.position = loca.vial_rack[1,1]
c9.set_output('gripper', False)


# move substrate from coating station to probe station
c9.tool = 'substrate_tool'
c9.position = loca.s_coater
c9.set_output('substrate_tool', True)
c9.set_output('coater_stage_vacuum', False)
proc.rotate_the_sample(c9, times=1, direction='CCW', sample_on_stage=False)
c9.position = loca.probe_seq
c9.set_output('substrate_tool', False)
c9.tool = None


# move vial from clamp to rack
c9.position = loca.clamp
c9.set_output('gripper', True)
c9.set_output('clamp', False)
c9.position = loca.cooking_rack[0,0]
c9.set_output('gripper', False)

# put PDMS stamp on top of the device
c9.tool = 'substrate_tool'
c9.position = loca.substrate_rack_seq[0,0]
c9.set_output('substrate_tool', True)
c9.position = loca.probe_seq
c9.set_output('substrate_tool', False)
c9.tool = None

# add NaCl
c9.reset_pump()
c9.tool = None
V_NaCl = 0.1
proc.move_vial_to_clamp(c9, ('vial', 'NaCl'))
proc.aspirate_clamp_solution(c9, V_NaCl)
c9.position = loca.p_PDMS
c9.dispense_ml(0, V_NaCl)
proc.remove_pipette(c9)
proc.move_vial_from_clamp(c9, 'vial')

# maunally load a sample json file
smp = ro.sample.loads('5d4a8e4679')
smp.done()  # transfer to the done directory

# test electrical measurement
Keithley = ro.system.init('IV')
Keithley.reset()
# c9.set_output('air_extention_12-17',True) #resetting keithley should already do this air extension step

probe_loc = {
    1: 11800 - 1750,  # 1st device
    2: 12800 - 1750,  # 2nd device
    3: 13800 - 1750,  # 3rd device
    4: 14700 - 1750,  # 4th device
    5: 15700 - 1750,  # 5th device
    6: 16700 - 1750,  # 6th device
}

# probe_loc = {
#         1: 10050,  # 1st device
#         2: 11350,  # 2nd device
#         3: 12350,  # 3rd device
#         4: 13250,  # 4th device
#         5: 14250,  # 5th device
#         6: 14950,  # 6th device
#     }

# probe_loc = {
#         1: 10050,  # 1st device
#         2: 11050,  # 2nd device
#         3: 12050,  # 3rd device
#         4: 12950,  # 4th device
#         5: 13950,  # 5th device
#         6: 14950,  # 6th device
#     }

c9.move_axis('pin_outer', 16000, vel=8000, accel=8000)
c9.move_axis('pin_inner', 13250, vel=8000, accel=8000)
c9.set_output('pin_inner', True)

output = {
        'forward_sweep': {},
        "reverse_sweep": {},
    }

c9.move_axis('pin_outer', 16000, vel=8000, accel=8000) # this moves the gate electrode in OECT device

for device in [2, 3, 4, 5]:
    c9.move_axis('pin_inner', probe_loc[device], vel=8000, accel=8000)
    c9.set_output('pin_inner', True)
    sleep(2)

for device in [3,4]:
    c9.move_axis('pin_inner', probe_loc[device], vel=8000, accel=8000)
    c9.set_output('pin_inner', True)
    sleep(2)
    ro.logger.info(f'measure IV of device {device}')
    output['forward_sweep'][
        f"device_{device}"
    ] = Keithley.retrieve_data_OECT()
    output['reverse_sweep'][
        f"device_{device}"
    ] = Keithley.retrieve_data_OECT(reverse=True)
    data = output['forward_sweep'][f'device_{device}']['I2']
    ro.logger.info(f"device_{device} I2: {data}")

c9.move_axis('pin_inner', 0, vel=8000, accel=8000)
c9.move_axis('pin_outer', 0, vel=8000, accel=8000)


# flip camera stage in
c9.set_output('camera_stage_a', True)
c9.set_output('camera_stage_b', False)

# flip camera stage out
c9.set_output('camera_stage_a', False)
c9.set_output('camera_stage_b', True)

# reload loca
from importlib import reload
reload(loca)


# ------- done for the day ------

import robotics as ro
ro.system.init('temperature').reset()

# turn off the LED light
# turn off N2 flow











# holding vial for pipette, out of soln

proc.move_vial_to_clamp(c9, ('cooking', sol_label))

# input('press any keys to continue')

coater.position = 45.5

# proc.aspirate_clamp_solution(c9, V, loca.p_clamp_20ul)
proc.new_pipette(c9)
# c9.position = loca.clamp
# c9.set_output('gripper', True)
# c9.uncap(pitch=1.75, revs=2, vel=5000, accel=5000)
# uncap_z_clamp = c9.position[3]  # uncap position
c9.position = loca.pipette_coater_20uL[:3] + [0]
input("move vial to pipette tip. press enter to continue")
c9.aspirate_ml(0, V)  # 1st arg is pump number
input("remove vial. press enter to continue")
#c9.goto(loca.clamp[:3] + [c9.position[3]], vel=10000)




















_____________
To get out of the interactive python in terminal -> 'raise SystemExit'

'settings.yml' 
# tells the Polybot which workflow to use

'python -m robotics' 
# robotics is the package for the backend of the Polybot, its like the jupyter notebook (the web server/app) for Polybot
# this command runs through the workflow and initializes all the hardware, it also defines all the functions but does not run them

'python -m robotics --sim' is for simulation more ctrl+C to quit after sample file is made

enter sample ID into 'measure_IV.py'

run in TERMINAL 'python .\measure_IV.py'






______
2/8/24 ~6:30pm
ERROR: uncaught exception in <Thread(Thread-7, started daemon 11472)>
  Traceback (most recent call last):
    File "C:\Users\Public\robot\polybot-env\lib\threading.py", line 975, in _bootstrap_inner
      self._invoke_excepthook(self)
    File "C:\Users\Public\robot\polybot-env\lib\threading.py", line 1247, in invoke_excepthook
      hook(args)
    File "c:\users\public\robot\robotics\src\robotics\__init__.py", line 120, in thread_excepthook
      excepthook(
    File "c:\users\public\robot\robotics\src\robotics\__init__.py", line 111, in excepthook
      trace_list = format_exception(e, s, tb)
    File "C:\Users\Public\robot\polybot-env\lib\threading.py", line 973, in _bootstrap_inner
      self.run()
    File "C:\Users\Public\robot\polybot-env\lib\threading.py", line 910, in run
      self._target(*self._args, **self._kwargs)
    File "c:\users\public\robot\robotics\src\robotics\scheduler.py", line 110, in start
      step.execute(smp)
    File "c:\users\public\robot\robotics\src\robotics\workflow\base.py", line 177, in execute
      out = func(smp, **kwargs)
    File "./workflow_OECT.py", line 186, in coating_on_top
      proc.move_vial_to_clamp(c9, ('cooking', sol_label))
    File "c:\users\public\robot\robotics\src\robotics\procedure.py", line 432, in move_vial_to_clamp
      c9.position = c9.loca.clamp
    File "c:\users\public\robot\robotics\src\robotics\_system\controller.py", line 152, in position
      return self.goto_safe(loc, vel=vel, accel=accel)
    File "c:\users\public\robot\robotics\src\robotics\_system\controller.py", line 401, in goto_safe
      return super().goto_safe(loc, vel=vel, accel=accel)
    File "C:\Users\Public\robot\polybot-env\lib\site-packages\north_c9\north_c9.py", line 588, in goto_safe
      self.goto_xy_safe(loc_list, vel=vel, accel=accel, wait=True)
    File "C:\Users\Public\robot\polybot-env\lib\site-packages\north_c9\north_c9.py", line 580, in goto_xy_safe
      self.move_robot_cts(loc_list[self.GRIPPER], loc_list[self.ELBOW], loc_list[self.SHOULDER], self.safe_height,
    File "c:\users\public\robot\robotics\src\robotics\_system\controller.py", line 390, in move_robot_cts
      return self.new_cmd_token(self.get_robot_status, self.FREE, wait=wait)
    File "C:\Users\Public\robot\polybot-env\lib\site-packages\north_c9\north_c9.py", line 728, in new_cmd_token
      tkn.wait()
    File "C:\Users\Public\robot\polybot-env\lib\site-packages\north_c9\north_c9.py", line 67, in wait
      while not self.is_done():
    File "C:\Users\Public\robot\polybot-env\lib\site-packages\north_c9\north_c9.py", line 54, in is_done
      result = self.wait_func()
      args = self.send_packet('ROST')
    File "C:\Users\Public\robot\polybot-env\lib\site-packages\north_c9\north_c9.py", line 358, in send_packet
      raise e
    File "C:\Users\Public\robot\polybot-env\lib\site-packages\north_c9\north_c9.py", line 354, in send_packet
      packet_len = self.network.read(1, 0.6)  # pumps need ~500ms timeout in FW, so this should be larger
    File "C:\Users\Public\robot\polybot-env\lib\site-packages\ftdi_serial.py", line 596, in read
      raise SerialReadTimeoutException('Read timeout')
  ftdi_serial.SerialReadTimeoutException: Read timeout




from shift_loc import cts_shiftxy
cts_shiftxy([68, 13481, 34895, 20800], shift=(0,0.5)) # mm 


camera = ro.system.init('camera')
ID = 'fe73T1'
smp = ro.sample.loads(ID)
fname = 'annealed_film_gel'
img_name, img = camera.take_image(
    f"samples/media/{smp['ID']}_raw_{fname}",
    focus=35,
    crop=[0, 900, 600, 1550],
    name_only=False,
)
smp['raw_outputs'][camera.outkey][fname] = img
smp.save()







# # --------------------------------------------------------------
# @ro.workflow.register(workflow=wf)
# def store_sample(smp):
#     """Return sample to rack and end the experiment"""

#     c9.tool = 'substrate_tool'
#     c9.position = loca.probe_PDMS_seq
#     seq, new_label = 'substrate_rack_PDMS_back_seq', 'sample_w_PDMS'
#     c9.set_output('substrate_tool', True)
#     proc.store_substrate(
#         c9,
#         seq=seq,
#         label='RFU_new',
#         updated_label=new_label,
#     )

#     c9.tool = None

# --------------------------------------------------------------
