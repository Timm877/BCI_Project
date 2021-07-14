from datetime import datetime
from pythonosc import dispatcher
from pythonosc import osc_server

ip = "0.0.0.0"
port = 5000
abs_waves = [-1,-1,-1,-1,-1]

# each file is 1 minute of the mental state.
# try to measure for 3 different persons

#filePath = 'Meditative/1.csv'
#filePath = 'Meditative/2.csv'
#filePath = 'Meditative/3.csv'

#filePath = 'Concentrated/1.csv'
#filePath = 'Concentrated/2.csv'
#filePath = 'Concentrated/3.csv'

filePath = './Neutral/1.csv'
#filePath = 'Neutral/2.csv'
#filePath = 'Neutral/3.csv'

recording = False
f = open (filePath,'w+')
f.write('TimeStamp,MSEC,Delta_TP9,Delta_AF7,Delta_AF8,Delta_TP10, Theta_TP9,Theta_AF7,Theta_AF8,Theta_TP10,Alpha_TP9,Alpha_AF7,Alpha_AF8,Alpha_TP10,Beta_TP9,Beta_AF7,Beta_AF8,Beta_TP10,Gamma_TP9,Gamma_AF7,Gamma_AF8,Gamma_TP10')

MSEC = 0
WAVE = 0
DATA_LINE = ""
def eeg_handler(address: str,*args):
    global recording, MSEC, DATA_LINE, WAVE
    if recording:
        for arg in args[1:]:
            DATA_LINE += (","+str(arg))
        WAVE += 1

        if WAVE == 5:
            dateTimeObj = datetime.now()
            timestampStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f") 
            f.write(timestampStr + "," + str(MSEC) + DATA_LINE)
            MSEC += 0.1
            DATA_LINE = ""
            WAVE = 0
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

if __name__ == "__main__":
    dispatcher = dispatcher.Dispatcher()

    dispatcher.map("/muse/elements/delta_absolute", eeg_handler,0)
    dispatcher.map("/muse/elements/theta_absolute", eeg_handler,1)
    dispatcher.map("/muse/elements/alpha_absolute", eeg_handler,2)
    dispatcher.map("/muse/elements/beta_absolute", eeg_handler,3)
    dispatcher.map("/muse/elements/gamma_absolute", eeg_handler,4)

    dispatcher.map("/Marker/*", marker_handler)

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Listening on UDP port "+str(port)+"\nSend Marker 1 to Start recording and Marker 2 to Stop Recording.")
    server.serve_forever()