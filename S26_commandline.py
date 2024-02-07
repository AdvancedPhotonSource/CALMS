# Enables command line scripting for HXN microscope operation
# start this with /APSshare/anaconda/x86_64/bin/ipython -i S26_commandline.py

import sys
import epics
import epics.devices
import time
import datetime
import numpy as np
import os 
import math
from IPython import get_ipython
ipython=get_ipython()

# Define motors
fomx = epics.Motor('26idcnpi:m10.')  # motor for moving focusing optics in the x direction, a positive value moves the motor in the outboard direction 
fomy = epics.Motor('26idcnpi:m11.') # motor for moving focusing optics in the y direction, a positive value moves the motor up in the lab frame 
fomz = epics.Motor('26idcnpi:m12.') # motor for moving focusing optics in the z direction, a positive value moves the motor further downstream 
samy = epics.Motor('26idcnpi:m17.') # motor for moving the sample in the y direction, a positive value moves the sample up in the lab frame
samth = epics.Motor('atto2:PIC867:1:m1.')
osax = epics.Motor('26idcnpi:m13.')
osay = epics.Motor('26idcnpi:m14.')
osaz = epics.Motor('26idcnpi:m15.')
condx = epics.Motor('26idcnpi:m5.')
attox = epics.Motor('atto2:m4.') # motor for moving the sample in the x direction, a positive value move the sample outboard if samth is at 0 degree. 
attoz = epics.Motor('atto2:m3.') # motor for moving the sample in the z direction, a positive value move the sample downstream if samth is at 0 degree.
objx = epics.Motor('26idcnpi:m1.')
xrfx = epics.Motor('26idcDET:m7.') # motor for moving the XRF detector in	


DCMenergy = epics.Motor("26idbDCM:sm8")
hybridx = epics.Device('26idcnpi:X_HYBRID_SP.', attrs=('VAL','DESC')) # motor for moving the x-ray beam in the x direction. A positive value moves the beam outboard.
hybridx.add_pv('26idcnpi:m34.RBV', attr='RBV')
hybridy  = epics.Device('26idcnpi:Y_HYBRID_SP.', attrs=('VAL','DESC')) # motor for moving the x-ray beam in the y direction. A positive value moves the beam up in the lab frame.
hybridy.add_pv('26idcnpi:m35.RBV', attr='RBV')
twotheta = epics.Motor('26idcSOFT:sm3.')
not_epics_motors = [hybridx.NAME, hybridy.NAME, twotheta.NAME]


def movr(motor,tweakvalue):
    """
    Move motor to absolute position, tweakvalue is in um
    

    """
    if motor in [fomx, fomy, samy]:
        epics.caput('26idcnpi:m34.STOP',1)
        epics.caput('26idcnpi:m35.STOP',1)
    if ( (motor in [hybridx, hybridy]) and ( (abs(hybridx.RBV-hybridx.VAL)>100) or (abs(hybridy.RBV-hybridy.VAL)>100) ) ):
        print("Please use lock_hybrid() to lock piezos at current position first...")
        return
    if motor.NAME in not_epics_motors:
        motor.VAL = motor.VAL+tweakvalue
        time.sleep(1)
        print(motor.DESC+"--->  "+str(motor.RBV))
    else:
        result = motor.move(tweakvalue, relative=True, wait=True)
        if result==0:
            time.sleep(0.5)
            print(motor.DESC+" ---> "+str(motor.RBV))
            fp = open(logbook,"a")
            fp.write(motor.DESC+" ---> "+str(motor.RBV)+"\n")
            fp.close()
        else:
            print("Motion failed")


def lock_hybrid():
    """
    Function that must be called before any motor movement when applied to a hybrid motor
    denoted by the hybrid prefix
    """
    
    tempx = hybridx.RBV
    time.sleep(1)
    mov(hybridx,tempx)
    time.sleep(1)
    tempy = hybridy.RBV
    time.sleep(1)
    mov(hybridy,tempy)
    time.sleep(1)

def unlock_hybrid():
    """
    Function that must be called after any motor movement when applied to a hybrid motor
    denoted by the hybrid prefix
    """
    
    tempx = hybridx.RBV
    tempy = hybridy.RBV
    print("before unlock: x = {0} and y = {1}".format(tempx, tempy))
    epics.caput('26idcnpi:m34.STOP',1)
    epics.caput('26idcnpi:m35.STOP',1)
    if ( (abs(fomx.RBV-optic_in_x)<100) and (abs(fomy.RBV-optic_in_y)<100) ):  
        mov(fomx,optic_in_x);
        mov(fomy,optic_in_y);
    time.sleep(1)
    tempx = hybridx.RBV
    tempy = hybridy.RBV
    print("after unlock: x = {0} and y = {1}".format(tempx, tempy))


def scan1d(motor,startpos,endpos,numpts,dettime, absolute=False):
    """
     # if absolute flag is set to True, scan a single motor from start position (startpos) to end position (endpos), with number of points (numpts) and count time in seconds (dettime) 
     # if absolute flag is set to False, scan a single motor from current position minux startpos to current position plus endpos, with number of points (numpts) and count time in seconds (dettime) 
     """
    pass


# add scan2d

# Prompt: I want to perform the same scan at different focuses. 
# Make it more casual over time