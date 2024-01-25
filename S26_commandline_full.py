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


def lock_hybrid():

    # lock in the interferrometer
    
    tempx = hybridx.RBV
    time.sleep(1)
    mov(hybridx,tempx)
    time.sleep(1)
    tempy = hybridy.RBV
    time.sleep(1)
    mov(hybridy,tempy)
    time.sleep(1)

def unlock_hybrid():

    # disable interferrometer lock in 
    
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


def set_zp_in():

    # set current position at zp_in position
    
    global optic_in_x, optic_in_y, optic_in_z
    print("ZP X focal position set to: "+str(fomx.RBV))
    optic_in_x = fomx.RBV
    print("ZP Y focal position set to: "+str(fomy.RBV))
    optic_in_y = fomy.RBV
    print("ZP Z focal position set to: "+str(fomz.RBV))
    optic_in_z = fomz.RBV
    save_config()
    

dets_list = []
# Turn on/off detectors and set exposure times
def detectors(det_list):

    # define the detectors to be triggered
    
    numdets = np.size(det_list)
    if(numdets<1 or numdets>4):
        print("Unexpected number of detectors")
    else:
        sc1.T1PV = ''
        sc1.T2PV = ''
        sc1.T3PV = ''
        sc1.T4PV = ''
        for ii in range(numdets):
            if det_list[ii]=='scaler':
                exec('sc1.T'+str(ii+1)+'PV = \'26idc:3820:scaler1.CNT\'')
            if det_list[ii]=='xrf':
                exec('sc1.T'+str(ii+1)+'PV = \'26idcXMAP:EraseStart\'')
            if det_list[ii]=='xrf_hscan':
                exec('sc1.T'+str(ii+1)+'PV = \'26idbSOFT:scanH.EXSC\'')
            if det_list[ii]=='andor':
                exec('sc1.T'+str(ii+1)+'PV = \'26idcNEO:cam1:Acquire\'')
            if det_list[ii]=='ccd':
                exec('sc1.T'+str(ii+1)+'PV = \'26idcCCD:cam1:Acquire\'')
            if det_list[ii]=='pixirad':
                exec('sc1.T'+str(ii+1)+'PV = \'dp_pixirad_xrd75:cam1:Acquire\'')
            #if det_list[ii]=='pilatus':
#                exec('sc1.T'+str(ii+1)+'PV = \'S18_pilatus:cam1:Acquire\'')
            #if det_list[ii]=='pilatus':
            #    exec('sc1.T'+str(ii+1)+'PV = \'dp_pilatusASD:cam1:Acquire\'')
            if det_list[ii]=='pilatus':
                exec('sc1.T'+str(ii+1)+'PV = \'dp_pilatus4:cam1:Acquire\'')
            #if det_list[ii]=='pilatus':
            #    exec('sc1.T'+str(ii+1)+'PV = \'S33-pilatus1:cam1:Acquire\'')
            if det_list[ii]=='medipix':
                exec('sc1.T'+str(ii+1)+'PV = \'QMPX3:cam1:Acquire\'')
                #exec('sc1.T'+str(ii+1)+'PV = \'dp_pixirad_msd1:cam1:MultiAcquire\'')
            if det_list[ii]=='vortex':
                exec('sc1.T'+str(ii+1)+'PV = \'dp_vortex_xrd77:mca1EraseStart\'')
	dets_list = det_list		

def count_time(dettime):

    # define the count time
    
    det_trigs = [sc1.T1PV, sc1.T2PV, sc1.T3PV, sc1.T4PV]
    if '26idc:3820:scaler1.CNT' in det_trigs:
        epics.caput("26idc:3820:scaler1.TP",dettime)
    #if ('26idcXMAP:EraseStart' in det_trigs) or ('26idbSOFT:scanH.EXSC' in det_trigs):
    epics.caput("26idcXMAP:PresetReal",dettime)
    if '26idcNEO:cam1:Acquire' in det_trigs:
        epics.caput("26idcNEO:cam1:Acquire",0)
        time.sleep(0.5)
        epics.caput("26idcNEO:cam1:AcquireTime",dettime)
        epics.caput("26idcNEO:cam1:ImageMode","Fixed")
    if '26idcCCD:cam1:Acquire' in det_trigs:
        epics.caput("26idcCCD:cam1:Acquire",0)
        time.sleep(0.5)
        epics.caput("26idcCCD:cam1:AcquireTime",dettime)
        epics.caput("26idcCCD:cam1:ImageMode","Fixed")
        time.sleep(0.5)
        epics.caput("26idcCCD:cam1:Initialize",1)
    if 'dp_pixirad_xrd75:cam1:Acquire' in det_trigs:
        epics.caput("dp_pixirad_xrd75:cam1:AcquireTime",dettime)
    #if 'dp_pilatusASD:cam1:Acquire' in det_trigs:
    #    epics.caput("dp_pilatusASD:cam1:AcquireTime",dettime)
    if 'dp_pilatus4:cam1:Acquire' in det_trigs:
        epics.caput("dp_pilatus4:cam1:AcquireTime",dettime)
    if 'QMPX3:cam1:Acquire' in det_trigs:
        epics.caput("QMPX3:cam1:AcquirePeriod",dettime*1000)
        #epics.caput("QMPX3:cam1:AcquirePeriod",500)
        #epics.caput("QMPX3:cam1:NumImages",np.round(dettime/0.5))
#    if 'S33-pilatus1:cam1:Acquire' in det_trigs:
#        epics.caput("S33-pilatus1:cam1:AcquireTime",dettime)
#    if 'S18_pilatus:cam1:Acquire' in det_trigs:
#        epics.caput("S18_pilatus:cam1:AcquireTime",dettime)
    # if 'dp_pixirad_msd1:MultiAcquire' in det_trigs:
   #     epics.caput("dp_pixirad_msd1:cam1:AcquireTime",dettime)
   # if 'dp_pixirad_msd1:cam1:Acquire' in det_trigs:
   #     epics.caput("dp_pixirad_msd1:cam1:AcquireTime",dettime)
    if 'dp_vortex_xrd77:mca1EraseStart' in det_trigs:
        epics.caput("dp_vortex_xrd77:mca1.PRTM",dettime)

def prescan():

    # pre scan macro
    
    scannum = epics.caget(scanrecord+':saveData_scanNumber',as_string=True)
    print("scannum is {0}".format(scannum))
    pathname = epics.caget(scanrecord+':saveData_fullPathName',as_string=True)
    detmode = epics.caget("QMPX3:cam1:ImageMode");
    savemode = epics.caget("QMPX3:TIFF1:EnableCallbacks")
    if( detmode == 2 ):
        print("Warning - Medipix is in continuous acquisition mode - changing this to single")
        epics.caput("QMPX3:cam1:ImageMode",0)
        time.sleep(1)
    if( savemode == 0):
        print("Warning - Medipix is not saving images - enabling tiff output")
        epics.caput("QMPX3:TIFF1:EnableCallbacks",1)
        time.sleep(1)
    if( epics.caget('PA:26ID:SCS_BLOCKING_BEAM.VAL') ):
        print("Warning - C station shutter is closed - opening shutter")
        epics.caput("PC:26ID:SCS_OPEN_REQUEST.VAL",1)
        time.sleep(2)
    epics.caput("QMPX3:TIFF1:FilePath",pathname[:-4]+'Images/'+scannum+'/')
    time.sleep(1)
    epics.caput("QMPX3:TIFF1:FileName",'scan_'+scannum+'_img')
    time.sleep(1)
    for i in range(1,5):
        det_name = epics.caget("26idbSOFT:scan1.T{0}PV".format(i))
        if 'pilatus' in det_name:
            epics.caput("dp_pilatus4:cam1:FilePath",'/home/det/s26data/'+pathname[15:-4]+'Images/'+str(scannum)+'/')
            time.sleep(1)
            epics.caput("dp_pilatus4:cam1:FileName",'scan_'+scannum+'_pil')
            time.sleep(1)
    epics.caput("26idc:filter:Fi1:Set",0)
    time.sleep(1)
    return 0

def postscan():

    # post scan macro
    
    pathname = epics.caget(scanrecord+':saveData_fullPathName',as_string=True)
    epics.caput("QMPX3:TIFF1:FilePath",pathname[:-4]+'Images/')
    time.sleep(1)
    epics.caput("QMPX3:TIFF1:FileName",'image')
    time.sleep(1)
    for i in range(1,5):
        det_name = epics.caget("26idbSOFT:scan1.T{0}PV".format(i))
        if 'pilatus' in det_name:
            epics.caput("dp_pilatus4:cam1:FilePath",'/home/det/s26data/'+pathname[15:-4]+'Images/')
            time.sleep(1)
            epics.caput("dp_pilatus4:cam1:FileName",'pilatus')
            time.sleep(1)
    epics.caput("26idc:filter:Fi1:Set",1)
    time.sleep(1)

 

def scan1d(motor,startpos,endpos,numpts,dettime, absolute=False):

    # scan 1 motor from start position (startpos) to end position (endpos), with number of points (numpts) and count time (dettime) 
    
    if motor in [fomx, fomy, samy]:
        epics.caput('26idcnpi:m34.STOP',1)
        epics.caput('26idcnpi:m35.STOP',1)
    if ( (motor in [hybridx, hybridy]) and ( (abs(hybridx.RBV-hybridx.VAL)>100) or (abs(hybridy.RBV-hybridy.VAL)>100) ) ):
        print("Please use lock_hybrid() to lock piezos at current position first...")
        return
    sc1.P1PV = motor.NAME+'.VAL'
    if absolute:
        sc1.P1AR=0
    else:
        sc1.P1AR=1
    sc1.P1SP = startpos
    sc1.P1EP = endpos
    sc1.NPTS = numpts
    count_time(dettime)
    fp = open(logbook,"a")
    fp.write(' ----- \n')
    fp.write('SCAN #: '+epics.caget(scanrecord+':saveData_scanNumber',as_string=True)+' ---- '+str(datetime.datetime.now())+'\n')
    if absolute:
        fp.write('Scanning '+motor.DESC+' from '+str(startpos)+' ---> '+str(endpos)+' in '+str(numpts)+' points at '+str(dettime)+' seconds acquisition\n')
    else:
        fp.write('Scanning '+motor.DESC+' from '+str(startpos+motor.VAL)+' ---> '+str(endpos+motor.VAL))
        fp.write(' in '+str(numpts)+' points at '+str(dettime)+' seconds acquisition\n')
    fp.write(' ----- \n')
    fp.close()
    time.sleep(1)
    stopnow = prescan();
    if (stopnow):
        return
    sc1.execute=1
    print("Scanning...")
    time.sleep(1)
    while(sc1.BUSY == 1):
        time.sleep(1)
    postscan()

def scan2d(motor1,startpos1,endpos1,numpts1,motor2,startpos2,endpos2,numpts2,dettime, absolute=False):

    # scan two motors for a mesh scan
    
    if (motor1 in [fomx, fomy, samy]) or (motor2 in [fomx, fomy, samy]):
        epics.caput('26idcnpi:m34.STOP',1)
        epics.caput('26idcnpi:m35.STOP',1)
    if ( ( (motor1 in [hybridx, hybridy]) or (motor2 in [hybridx,hybridy]) ) and ( (abs(hybridx.RBV-hybridx.VAL)>100) or (abs(hybridy.RBV-hybridy.VAL)>100) ) ):
        print("Please use lock_hybrid() to lock piezos at current position first...")
        return
    sc2.P1PV = motor1.NAME+'.VAL'
    sc1.P1PV = motor2.NAME+'.VAL'
    if absolute:
        sc1.P1AR=0
        sc2.P1AR=0
    else:
        sc1.P1AR=1
        sc2.P1AR=1
    sc2.P1SP = startpos1
    sc1.P1SP = startpos2
    sc2.P1EP = endpos1
    sc1.P1EP = endpos2
    sc2.NPTS = numpts1
    sc1.NPTS = numpts2
    count_time(dettime)
    fp = open(logbook,"a")
    fp.write(' ----- \n')
    fp.write('SCAN #: '+epics.caget(scanrecord+':saveData_scanNumber',as_string=True)+' ---- '+str(datetime.datetime.now())+'\n')
    if absolute:
        fp.write('2D Scan:\n')
        fp.write('Inner loop: '+motor2.DESC+' from '+str(startpos2)+' ---> '+str(endpos2))
        fp.write(' in '+str(numpts2)+' points at '+str(dettime)+' seconds acquisition\n')
        fp.write('Outer loop: '+motor1.DESC+' from '+str(startpos1)+' ---> '+str(endpos1))
        fp.write(' in '+str(numpts1)+' points at '+str(dettime)+' seconds acquisition\n')   
    else:
        fp.write('2D Scan:\n')
        fp.write('Outer loop: '+motor1.DESC+' from '+str(startpos1+motor1.VAL)+' ---> '+str(endpos1+motor1.VAL))
        fp.write(' in '+str(numpts1)+' points at '+str(dettime)+' seconds acquisition\n')
        fp.write('Inner loop: '+motor2.DESC+' from '+str(startpos2+motor2.VAL)+' ---> '+str(endpos2+motor2.VAL))
        fp.write(' in '+str(numpts2)+' points at '+str(dettime)+' seconds acquisition\n')
    fp.write(' ----- \n')
    fp.close()
    time.sleep(1)
    stopnow = prescan();
    if (stopnow):
        return
    sc2.execute=1
    print("Scanning...")
    time.sleep(1)
    while(sc2.BUSY == 1):
        time.sleep(1)
    postscan()

def focalseries(z_range,numptsz,y_range,numptsy,dettime,motor1=fomz,motor2=hybridy):

    # a 2D scan with the zone plate position (fomz) as one of the motors, to determine if sample is at focus
    
    sc1.P1PV = motor2.NAME+'.VAL'
    sc2.P1PV = motor1.NAME+'.VAL'
    sc1.P1SP = -y_range/2.0
    sc2.P1SP = -z_range/2.0
    sc1.P1EP = y_range/2.0
    sc2.P1EP = z_range/2.0
    sc1.NPTS = numptsy
    sc2.NPTS = numptsz
    sc1.P1AR = 1
    sc2.P1AR = 1
    sc2.P2AR = 1
    sc2.P3AR = 1
    sc2.P2PV = hybridy.NAME+'.VAL'
    sc2.P2SP = 1.177*z_range/400   #change y offset here
    sc2.P2EP = -1.177*z_range/400
    sc2.P3PV = hybridx.NAME+'.VAL'
    sc2.P3SP = 0.3125*z_range/400   #change x offset here
    sc2.P3EP = -0.3125*z_range/400
    count_time(dettime)
    time.sleep(1)
    if ( (abs(hybridx.RBV-hybridx.VAL)>50) or (abs(hybridy.RBV-hybridy.VAL)>50) ):
        print("Please use lock_hybrid() to lock piezos at current position first...")
        sc2.P2PV = ''
        sc2.P3PV = ''
        return
    stopnow = prescan();
    if (stopnow):
        return
    sc2.execute=1
    print("Scanning...")
    time.sleep(1)
    while(sc2.BUSY == 1):
        time.sleep(1)
    postscan()
    time.sleep(2)
    sc2.P2PV = ''
    sc2.P3PV = ''


def timeseries(numpts,dettime=1.0):

    # a time series, taking numpts acquisitions at given count time (dettime)
    
    tempsettle1 = sc1.PDLY
    tempsettle2 = sc1.DDLY
    tempdrive = sc1.P1PV
    tempstart = sc1.P1SP
    tempend = sc1.P1EP
    sc1.PDLY = 0.0
    sc1.DDLY = 0.0
    sc1.P1PV = "26idcNES:sft01:ph01:ao03.VAL"
    sc1.P1AR = 1
    sc1.P1SP = 0.0
    sc1.P1EP = numpts*dettime
    sc1.NPTS = numpts+1
    count_time(dettime)
    fp = open(logbook,"a")
    fp.write(' ----- \n')
    fp.write('SCAN #: '+epics.caget(scanrecord+':saveData_scanNumber',as_string=True)+' ---- '+str(datetime.datetime.now())+'\n')
    fp.write('Timeseries: '+str(numpts)+' points at '+str(dettime)+' seconds acquisition\n')
    fp.write(' ----- \n')
    fp.close()
    time.sleep(1)
    stopnow = prescan();
    if (stopnow):
        return
    sc1.execute=1
    print("Scanning...")
    time.sleep(2)
    while(sc1.BUSY == 1):
        time.sleep(1)
    postscan()
    sc1.PDLY = tempsettle1
    sc1.DDLY = tempsettle2
    sc1.P1PV = tempdrive
    sc1.P1SP = tempstart
    sc1.P1EP = tempend

 
def spiralsquare(spiral_step, spiral_ctime):

    # using the look up table to perform a customized spiral scan

    print("if you abort this scan, please make sure the scanmode is switched back and sc1.P2PV cleared !")
    # add this to my cleanup macro so that it is done automatically in the future

    if abs(hybridx.RBV-hybridx.VAL)>100 or abs(hybridy.RBV-hybridy.VAL)>100:
        print("Please use lock_hybrid() to lock piezos at current position first...")
        return

    sc1.P1PV = "26idcnpi:X_HYBRID_SP.VAL"
    sc1.P2PV = "26idcnpi:Y_HYBRID_SP.VAL"

    sc1.P1AR = 0  # absolute, not sure it is useful, but be safe
    sc1.P2AR = 0  # absolute, not sure it is useful, but be safe

    spiral_x0 = hybridx.RBV
    spiral_y0 = hybridy.RBV 

    spiral_traj = np.load("optimized_route.npz")

    spiral_npts = int(spiral_traj['x'].shape[0])

    spiral_x = spiral_traj['x']*spiral_step+spiral_x0
    spiral_y = spiral_traj['y']*spiral_step+spiral_y0
    count_time(spiral_ctime)
    
    sc1.NPTS = spiral_npts

    sc1.P1PA = spiral_x
    sc1.P2PA = spiral_y
    print("switching to look up mode")
    sc1.P1SM = 1
    sc1.P2SM = 1
    time.sleep(1)

    stopnow = prescan();
    if (stopnow):
        return
    sc1.execute=1
    print("Scanning...")
    time.sleep(1)
    while(sc1.BUSY == 1):
        time.sleep(1)
    postscan()

    print("switching to linear mode")
    sc1.P1SM = 0
    sc1.P2SM = 0
    sc1.P2PV = ""



