
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
fomx = epics.Motor('26idcnpi:m10.')  # motor for moving focusing optics in the x direction, a positive value moves the motor in the outboard direction (will it understand a negative value is then inboard???)
fomy = epics.Motor('26idcnpi:m11.') # motor for moving focusing optics in the y direction, a positive value moves the motor up in the lab frame (will it understand a negative value is then down???)
fomz = epics.Motor('26idcnpi:m12.') # motor for moving focusing optics in the z direction, a positive value moves the motor further downstream (will it understand a negative value is then upstream???)
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



def mov(motor,position):

    # move motor to absolute position
    
    if motor in [fomx, fomy, samy]:
        epics.caput('26idcnpi:m34.STOP',1)
        epics.caput('26idcnpi:m35.STOP',1)
        epics.caput('26idcSOFT:userCalc1.SCAN',0)
        epics.caput('26idcSOFT:userCalc3.SCAN',0)
    if motor.NAME in not_epics_motors:
        motor.VAL = position
        time.sleep(1)
        print(motor.DESC+"--->  "+str(motor.RBV))
    else:
        result = motor.move(position, wait=True)
        if result==0:
            time.sleep(0.5)
            print(motor.DESC+" ---> "+str(motor.RBV))
            fp = open(logbook,"a")
            fp.write(motor.DESC+" ---> "+str(motor.RBV)+"\n")
            fp.close()
            epics.caput('26idcSOFT:userCalc1.SCAN',6)
            epics.caput('26idcSOFT:userCalc3.SCAN',6)
        else:
            print("Motion failed")
        
    
def movr(motor,tweakvalue):

    # move motor by tweakvalue
    
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

def zp_in():

    # move in the zone plate to allow focused beam
    
    print('Moving ZP to focal position...\n')
    epics.caput('26idcSOFT:userCalc1.SCAN',0);
    epics.caput('26idcSOFT:userCalc3.SCAN',0);
    epics.caput('26idbSOFT:userCalc3.SCAN',0);
    mov(fomx,optic_in_x)
    mov(fomy,optic_in_y)
    mov(fomz,optic_in_z)
    epics.caput('26idcSOFT:userCalc1.SCAN',5);
    epics.caput('26idcSOFT:userCalc3.SCAN',5);
    epics.caput('26idbSOFT:userCalc3.SCAN',5);

def zp_out():

    # move out the zone plate to allow parallel beam
    
    tempx = epics.caget('26idc:sft01:ph02:ao09.VAL')
    tempy = epics.caget('26idc:robot:Y1.VAL')
    temp2th = epics.caget('26idcDET:base:Theta.VAL')
    if ( (abs(mpx_in_x-tempx)<0.1) and (abs(mpx_in_y-tempy)<0.1) and (abs(temp2th)<1.0) ):
        print("Please use genie_in() to move medipix out of beam first...")
        return
    print('Moving ZP out of beam...\n')
    epics.caput('26idcSOFT:userCalc1.SCAN',0);
    epics.caput('26idcSOFT:userCalc3.SCAN',0);
    epics.caput('26idbSOFT:userCalc3.SCAN',0);
    mov(fomx,optic_in_x+3500.0)
    mov(fomy,optic_in_y)
    mov(fomz,-4700.0)
    epics.caput('26idcSOFT:userCalc1.SCAN',5);
    epics.caput('26idcSOFT:userCalc3.SCAN',5);
    epics.caput('26idbSOFT:userCalc3.SCAN',5);
