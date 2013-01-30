#
# Audio Visualizer RGB lights, v1.0
# Author: Jared Sanson
# Created: 11/01/2013
#
# LED strip will be a blue colour when the room is calm,
# and progressively turn to red as the room gets rowdy.
# 
# The settings below should work fine,
# try adjusting the microphone volume/gain before you touch them!
#
# Connects to an arduino, which controls the RGB led strip.
#
# Required libraries:
# Python 2.7
# SciPy      : http://www.scipy.org/
# Console    : http://effbot.org/downloads/#console
# pySerial   : http://pyserial.sourceforge.net/
# pyAudio    : http://people.csail.mit.edu/hubert/pyaudio/
#

########## LIBRARIES ###########################################################

import Console
import scipy.fftpack
import pyaudio
import time
import struct
import scipy
import math
from scipy import pi, signal
from scipy.fftpack import fft,rfft,rfftfreq,fftfreq

import serial
import struct
import time


########## VISUALIZATION SETTINGS ##############################################

# Loudness detect:
CHANNEL = 1     # frequency channel of the FFT to use (see console output to decide)
GAIN = 1.5       # audio gain (multiplier)
THRESHOLD = 0.15 # audio trigger threshold

#
ATTACK = 0.004  # amount of rowdz increase with loudness
DECAY = 0.003   # amount of rowdz decay

# Brightness:
MODULATION = 0.0        # amount of loudness flickering modulation
MIN_BRIGHTNESS = 0.5    # minimum brightness

# Hue mapping:
MIN_HUE = 200   # Aqua
MAX_HUE = 0     # Red
# Note that the hue mapping is actually a power function,
# so it will spend more time towards the MIN_HUE, and only a short time towards the MAX_HUE.

########## APPLICATION SETTINGS ################################################

#COM_PORT = 'COM3'   # COM port to use, or None to run without an arudino
COM_PORT = None

# Audio capture settings
SAMPLE_RATE = 44100
BUFFER_SIZE = 2**11     # Changing this will change the frequency response of the algorithm
#CUTOFF_FREQ = 20000     # LPF freq (Hz)


########## SUPPORT FUNCTIONS ###################################################

def tobyte(i):
    """
    Clip values that fall outside an unsigned byte
    """
    i = int(i)
    if i < 0: i = 0
    if i > 255: i = 255
    return i
    
def limit(val, vmin, vmax):
    """
    Clip values that fall outside vmin & vmax
    """
    if val < vmin: return vmin
    if val > vmax: return vmax
    return val

def mapval(val, minin, maxin, minout, maxout):
    """
    Linear value mapping between in and out
    """
    norm = (val-minin)/(maxin-minin)
    return norm*(maxout-minout) + minout
    
def thresh(val, threshold):
    """
    A bit hard to describe, but this will return 0 
    when val is below the threshold, and will
    linearly map val to anything higher than threshold.

    The effect being that above the threshold, louder
    signals will have more of an effect.
    """
    val -= threshold
    if val < 0: val = 0
    val *= (1.0/threshold)
    return val
    
def hsv2rgb(h, s, v):
    """
    Convert H,S,V to R,G,B
    H: 0.0 to 360.0, S&V: 0.0 to 1.0
    R,G,B: 0 to 255
    """
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0: r, g, b = v, t, p
    elif hi == 1: r, g, b = q, v, p
    elif hi == 2: r, g, b = p, v, t
    elif hi == 3: r, g, b = p, q, v
    elif hi == 4: r, g, b = t, p, v
    elif hi == 5: r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)    
    return r, g, b
    
def get_fft(data):
    """
    Run the sample through a FFT, and normalize
    """
    FFT = fft(data)
    freqs = fftfreq(BUFFER_SIZE*2, 1.0/SAMPLE_RATE)
    #y = 20*scipy.log10(abs(FFT))/ 100

    y = abs(FFT[0:len(FFT)/2])/1000
    y = scipy.log(y) - 2
    return (freqs,y)
    
ffts=[]
def smoothMemory(ffty,degree=3):
    """
    Average samples. Taken from Python FFT tutorial
    """
    global ffts
    ffts = ffts+[ffty]
    if len(ffts) <=degree: return ffty
    ffts=ffts[1:]
    return scipy.average(scipy.array(ffts),0)
    

bar_len = 70
def update_bars(x,y):
    """
    Display a bar graph in the console
    """
    for i,_ in enumerate(y):
        a = int(min(max(y[i],0),1)*bar_len)

        label = str(x[i])[:5]
        label = ' '*(5-len(label)) + label

        text =  label +'[' + ('#'*a) + (' '*(bar_len-a)) + ']' + str(i)
        console.text(0, i+3, text)


########## SUPPORT CLASSES #####################################################

class RGBController(object):
    """
    Communicates with the Arduino
    """
    def __init__(self, port):
        self.ser = serial.Serial(port)
        time.sleep(1)

    def update(self, color):
        #print "Update color to (R:%d G:%d B:%d)" % (color[0], color[1], color[2])
        r = tobyte(color[0])
        g = tobyte(color[1])
        b = tobyte(color[2])
        packet = struct.pack('cBBB', 'X', r,g,b)
        #print packet
        self.ser.write(packet)


########## CODE ################################################################

TITLE = "Audio Visualizer"
INK = 0x1f

# Set up console
console = Console.getconsole(0)
console.title(TITLE)
console.text(0, 0, chr(0xfe) + ' ' + TITLE + ' ' + chr(0xfe))
console.text(0, 1, chr(0xcd) * 80)

# Create LPF filter
# norm_pass = 2*math.pi*CUTOFF_FREQ/SAMPLE_RATE
# norm_stop = 1.5*norm_pass
# (N, Wn) = signal.buttord(wp=norm_pass, ws=norm_stop, gpass=2, gstop=30, analog=0)
# (b, a) = signal.butter(N, Wn, btype='low', analog=0, output='ba')
# b *= 1e3


# Open Audio stream (uses default audio adapter)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16,channels=1,\
                rate=SAMPLE_RATE,input=True,output=False,frames_per_buffer=BUFFER_SIZE)
        
# Open comms port to Arduino
if COM_PORT:
    RGB = RGBController(COM_PORT)


########## GLOBAL VARIABLES ####################################################

red = 0
green = 0
blue = 0

noisiness = 0       # Noisiness level

########## VISUALIZATION LOOP ##################################################

while True:
    ## Part 1: Sample Audio ##

    # Get audio sample
    buf = stream.read(BUFFER_SIZE)
    data = scipy.array(struct.unpack("%dh"%(BUFFER_SIZE),buf))

    ## Part 2: Perform FFT and Filtering ##

    # Filter incoming data
    #data = signal.lfilter(b,a,data)

    # Generate FFT
    freqs,y = get_fft(data)

    # Average the samples
    #y=smoothMemory(y,3)

    # Normalize
    y = y / 5

    # Average into chunks of N
    N = 25
    yy = [scipy.average(y[n:n+N]) for n in range(0, len(y), N)]
    yy = yy[:len(yy)/2] # Discard half of the samples, as they are mirrored


    ## Part 3: Algorithm ##

    # Loudness detection
    loudness = thresh(yy[CHANNEL] * GAIN, THRESHOLD)

    # Noisiness meter
    noisiness -= DECAY
    noisiness += loudness * ATTACK
    noisiness = limit(noisiness, 0.0, 1.0)

    # Brightness modulation
    modulation = MODULATION * limit(noisiness, 0.0, 1.0)
    brightness = limit(MIN_BRIGHTNESS + (loudness * modulation), 0.0, 1.0)

    # Hue modulation (power relationship)
    mapping = 1.1 - (10 ** (1.0 - limit(noisiness, 0.0, 1.0)) / 10.0)
    mapping = mapval(mapping, 0.1, 1.0, 0.0, 1.0)
    hue = mapval(mapping, 0.0, 1.0, MIN_HUE, MAX_HUE)

    # Display colour
    red,green,blue = hsv2rgb(hue,1.0,brightness)
    if COM_PORT:
        RGB.update([int(red),int(green),int(blue)])

    # Debug information
    labels = list(yy)
    bars = list(yy)
    labels.extend(   ['-', 'loud','noise','map', 'brght', '-', 'hue','red','grn','blue'])
    bars.extend(     [0, loudness, noisiness, mapping, brightness, 0, hue/360.0, red/255.0,green/255.0,blue/255.0])

    update_bars(labels,bars)
    
########## #####################################################################
