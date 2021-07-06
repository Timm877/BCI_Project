from datetime import datetime
from pythonosc import dispatcher
from pythonosc import osc_server

ip = "0.0.0.0"
port = 5000
abs_waves = [-1,-1,-1,-1,-1]

#each file is 1 minute of the mental state.
# try to measure for 3 different persons

#filePath = 'Meditative/1.csv'
#filePath = 'Meditative/2.csv'
#filePath = 'Meditative/3.csv'

#filePath = 'Concentrated/1.csv'
#filePath = 'Concentrated/2.csv'
#filePath = 'Concentrated/3.csv'

filePath = 'Neutral/1.csv'
#filePath = 'Neutral/2.csv'
#filePath = 'Neutral/3.csv'



recording = False
f = open (filePath,'w+')
f.write('TimeStamp,RAW_TP9,RAW_AF7,RAW_AF8,RAW_TP10,AUX,Marker\n')

def eeg_handler(address: str,*args):
    global recording
    if recording:
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")
        f.write(timestampStr)
        for arg in args:
            f.write(","+str(arg))
        f.write("\n")
    
def marker_handler(address: str,i):
    global recording
    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")
    markerNum = address[-1]
    f.write(timestampStr+",,,,/Marker/"+markerNum+"\n")
    if (markerNum=="1"):        
        recording = True
        print("Recording Started.")
    if (markerNum=="2"):
        f.close()
        server.shutdown()
        print("Recording Stopped.")    

def abs_handler(address: str,*args):
    global abs_waves
    wave = args[0][0]
    if (len(args)==2): #If OSC Stream Brainwaves = Average Only
        abs_waves[wave] = args[1] #Single value for all sensors, already filtered for good data

if __name__ == "__main__":
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/eeg", eeg_handler) #already a notch filter is applied I think? Let's check later at the FFT.

   # We also load in the different waves which are filtered by the app already
   # dispatcher.map("/muse/elements/delta_absolute", abs_handler,0)
   # dispatcher.map("/muse/elements/theta_absolute", abs_handler,1)
   # dispatcher.map("/muse/elements/alpha_absolute", abs_handler,2)
   # dispatcher.map("/muse/elements/beta_absolute", abs_handler,3)
   # dispatcher.map("/muse/elements/gamma_absolute", abs_handler,4)

    #last we load in the marker for recording time
    dispatcher.map("/Marker/*", marker_handler)

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Listening on UDP port "+str(port)+"\nSend Marker 1 to Start recording and Marker 2 to Stop Recording.")
    server.serve_forever()