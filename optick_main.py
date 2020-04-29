import os
import math
import time
import busio
import board
import RPi.GPIO as GPIO
import sys
sys.path.append("/home/pi/.local/lib/python3.7/site-packages")

import numpy as np
from scipy.interpolate import griddata
from colour import Color

import adafruit_amg88xx
from CCS811_RPi import CCS811_RPi

import tensorflow as tf
from PIL import Image,ImageDraw,ImageOps
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import matplotlib.pyplot as plt 
import matplotlib.image as img 
from luma.core.interface.serial import i2c
from luma.core.render import canvas
from luma.oled.device import sh1106

detectChange = 20

SPICLK = 11
SPIMISO = 9
SPIMOSI = 10
SPICS = 8
mq7_dpin = 26
mq7_apin = 0
ccs811 = CCS811_RPi()
coppm = 0
co2ppm = 0
vocppb = 0
INITIALBASELINE = False
pause = 10

MINTEMP = 25 #low range of the sensor (this will be blue on the screen)
MAXTEMP = 31 #high range of the sensor (this will be red on the screen)
IGTEMP = 35 #beyond max range (values above this will be ignored)
cd = 6 #how many color values we can have

tfmodel = tf.keras.models.load_model('/home/pi/optick/model_new.h5')
serial = i2c(port=1, address=0x3C)
device = sh1106(serial)
path = "/home/pi/optick/images/screenshot.jpeg"
i2c_bus = busio.I2C(board.SCL, board.SDA)
sensor = adafruit_amg88xx.AMG88XX(i2c_bus)

hwid = ccs811.checkHWID()

ccs811.configureSensor(0b100000)
print('---------------------------------')
if(INITIALBASELINE > 0):
        ccs811.setBaseline(INITIALBASELINE)
        print(ccs811.readBaseline())

points = [(math.floor(ix / 8), (ix % 8)) for ix in range(0, 64)]
grid_x, grid_y = np.mgrid[0:7:64j, 0:7:64j]

#the list of colors we can choose from
blue = Color("indigo")
colors = list(Color("black").range_to(Color("white"), cd))

#create the array of colors
colors = [(int(c.red * 255), int(c.green * 255), int(c.blue * 255)) for c in colors]



def init():
    GPIO.setwarnings(False)
    GPIO.cleanup()         #clean up at the end of your script
    GPIO.setmode(GPIO.BCM)     #to specify whilch pin numbering system
    # set up the SPI interface pins
    GPIO.setup(SPIMOSI, GPIO.OUT)
    GPIO.setup(SPIMISO, GPIO.IN)
    GPIO.setup(SPICLK, GPIO.OUT)
    GPIO.setup(SPICS, GPIO.OUT)
    GPIO.setup(mq7_dpin,GPIO.IN,pull_up_down=GPIO.PUD_DOWN)
    GPIO.setup(detectChange, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.add_event_detect(detectChange,GPIO.RISING,bouncetime=500)

def constrain(val, min_val, max_val,ig_val):
    if val>ig_val or val<min_val:
        return min_val
    else:
        if val>max_val:
            return max_val
        else:
            return val

def map_value(x, in_min, in_max, out_min, out_max):
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

def calibrate():
    print("Calibrating..")
    max_list = []
    for row in sensor.pixels:
        row_max = max(row)
        max_list.append(row_max)
    full_max=max(max_list)
    if(full_max < MAXTEMP):
        return full_max
    else:
        return MINTEMP
    
    
def readadc(adcnum, clockpin, mosipin, misopin, cspin):
    if ((adcnum > 7) or (adcnum < 0)):
        return -1
    GPIO.output(cspin, True)    
    
    GPIO.output(clockpin, False)  # start clock low
    GPIO.output(cspin, False)     # bring CS low

    commandout = adcnum
    commandout |= 0x18  # start bit + single-ended bit
    commandout <<= 3    # we only need to send 5 bits here
    for i in range(5):
        if (commandout & 0x80):
            GPIO.output(mosipin, True)
        else:
            GPIO.output(mosipin, False)
        commandout <<= 1
        GPIO.output(clockpin, True)
        GPIO.output(clockpin, False)

    adcout = 0
    # read in one empty bit, one null bit and 10 ADC bits
    for i in range(12):
        GPIO.output(clockpin, True)
        GPIO.output(clockpin, False)
        adcout <<= 1
        if (GPIO.input(misopin)):
            adcout |= 0x1

    GPIO.output(cspin, True)
        
    adcout >>= 1       # first bit is 'null' so drop it
    return adcout

def mode(detectChange, data):
    time.sleep(1)
    data^=1
    start_time = time.perf_counter()
    while(time.perf_counter()-start_time<2):
        if(GPIO.event_detected(detectChange)):
            data = 2
            break
    if(data == 0):
        #print(data)
        air_quality()
    elif(data == 1):
        human_detection()
    else:
        data = 1
        print("Calibrating...")
        with canvas(device) as draw:
            draw.text((4, 28), "CALIBRATING...", fill="white")
        human_detection()
    return data
 
def air_quality():
    totalco = 0
    totalco2 =  0
    co2fn = 0
    totalvoc = 0
    basetime=time.perf_counter()
    print("AIR QUALITY MODE")
    print("please wait...")
    with canvas(device) as draw:
            draw.text((16, 28), "AIR QUALITY MODE", fill="white")
    while True:
        if(GPIO.event_detected(detectChange)):
            break
        COlevel=readadc(mq7_apin, SPICLK, SPIMOSI, SPIMISO, SPICS)
        print_val = ""
        if GPIO.input(mq7_dpin):
            with canvas(device) as draw:
                draw.text((0, 0), "CO not leak", fill="white")
            print("CO not leak")
            time.sleep(0.5)
        else:
            if COlevel != 0:
                coppm=19.32*((5-((COlevel/1024.0)*5))/((COlevel/1024.0)*5))**(-0.64)
            else:
                coppm=0
            print("CO: " +str("%.2f"%coppm)+" ppm")
            print_val += "CO:" +str("%.2f"%coppm)+" ppm"
        humidity = 50.00
        temperature = 25.00
        
        statusbyte = ccs811.readStatus()
        error = ccs811.checkError(statusbyte)
        if(error):
                print('ERROR:',ccs811.checkError(statusbyte))
                
        if(not ccs811.checkDataReady(statusbyte)):
                print('No new samples are ready')
                print('---------------------------------')
                time.sleep(pause)
                continue;
        result = ccs811.readAlg();
        if(not result):
                time.sleep(pause)
                continue;
        baseline = ccs811.readBaseline()
        co2ppm=result['eCO2']
        vocppb=result['TVOC']
        if coppm>15:
            totalco += coppm
        if co2ppm>1000:
            totalco2 += co2ppm
        totalvoc += vocppb
        currtime=time.perf_counter()-basetime
        co2fn=21000-2800*(math.tan((currtime/12+1310)/800))
        co2val=co2fn*currtime/12
        
        #print(totalco2,co2val,currtime)
        print('eCO2: ',co2ppm,' ppm')
        print_val += '\neCO2: ' + str(co2ppm) + ' ppm'
        print('TVOC: ',vocppb, 'ppb')
        print_val += '\nTVOC: ' + str(vocppb) + ' ppb'
        print('Last error ID: ',result['errorid'])
        
        
        if totalco>160000 or coppm>6400:
            print_val += "\nDANGEROUS CO"
            print('DANGEROUS CO')
        elif totalco2>co2val or co2ppm>100000:
            print_val += "\nDANGEROUS CO2"
            print('DANGEROUS CO2')
        elif totalvoc>675000 or vocppb>6000:
            print_val += "\nDANGEROUS VOC"
            print('DANGEROUS VOC')
        else:
            print_val += "\nSafe"
            print('SAFE')
        print('---------------------------------')

        with canvas(device) as draw:
            draw.text((0, 0), print_val, fill="white")
        time.sleep(1)

def human_detection():
    
        print("HUMAN DETECTION MODE")
        with canvas(device) as draw:
            draw.text((4, 28), "HUMAN DETECTION MODE", fill="white")
        MINTEMP = calibrate()
        with canvas(device) as draw:
            draw.text((4, 28), "PLEASE FACE THE WALL", fill="white")
        while True:
            if(GPIO.event_detected(detectChange)):
                break
            #read the pixels
            
            pixels = []
            for row in sensor.pixels:
                pixels = pixels + row
            pixels = [map_value(p, MINTEMP, MAXTEMP, 0, 2*cd-1) for p in pixels] #1024 for 4 degrees

            #perform interpolation
            bicubic = griddata(points, pixels, (grid_x, grid_y), method='cubic')

            #draw everything
            colorpix = []
            for ix, row in enumerate(bicubic):
                colorpix_row = []
                for jx, pixel in enumerate(row):
                    colorpix_array = colors[constrain(int(pixel), 0, cd-1, 2*cd-1)]
                    colorpix_row.append(colorpix_array)
                colorpix.append(colorpix_row)    
            pixarray = np.array(colorpix, dtype=np.uint8)
            im = Image.fromarray(pixarray)
            im.save(path,'JPEG',quality=90)
            if os.path.isfile(path):
                time.sleep(1)
                im = Image.open(path)
                im = im.resize((32,32), Image.ANTIALIAS)
                im = im.rotate(270)
                im = ImageOps.mirror(im)
                im = ImageOps.grayscale(im)
                im.save(path , 'JPEG', quality=90)     
                dataset = np.ndarray(shape=(1, 32, 32, 3),dtype=np.float32)
                image = load_img(path) # this is a PIL image
                image.thumbnail((32, 32))
                # Convert to Numpy Array
                x = img_to_array(image)
                dataset[0] = x
                img = np.expand_dims(x, 0)
                label=['Negative','Positive']
                ynew = tfmodel.predict_classes(img)
                print(label[int(ynew)])
                imag = Image.open(path) 
                imag = imag.resize((64, 64))
                if label[int(ynew)]:
                    with canvas(device,dither=True) as draw:
                        draw.bitmap((0,0), imag)
                        draw.text((72,28), label[int(ynew)], fill="white")
            time.sleep(0.1)

def main():
    init()
    data = 1
    while True:
        data = mode(detectChange, data) 


if __name__ =='__main__':
    try:
        main()
        pass
    except KeyboardInterrupt:
        pass

GPIO.cleanup()
