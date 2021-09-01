# work with streaming data
# see if I can get it in a pandas DF
# I should collect for 3 sec, then  do outlier detection, and design all features, and predict with RF model?
# then every 1 second a prediction should be made about the previous 3 seconds 
# if data is 3 sec: do stuff and predict, then remove first second, then again add new data untill 3 sec are full again, and predict once more
# then later try to make it a video game or sumth'ng
# https://github.com/Sentdex/BCI/blob/master/testing_and_making_data.py

# step 1: receive data from streaming
from pythonosc import dispatcher
from pythonosc import osc_server
import random
import pandas as pd
import copy
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import numpy as np
from Step_2_preprocess.OutlierDetection import DistributionBasedOutlierDetection
from Step_3_feature_engineering.Dim_reduction import PrincipalComponentAnalysis, IndependentComponentAnalysis
from Step_3_feature_engineering.TemporalAbstraction import NumericalAbstraction
from Step_3_feature_engineering.FrequencyAbstraction import FourierTransformation
from collections import defaultdict
import datetime as dt
import pickle

#ALL GLOBAL VARIABLES:
hsi = [4,4,4,4]
hsi_string = ""

#Note: The names of the cols from muse device do not align with names in earlier files (Fp1 instead of AF7 and Fp2 instead of AF8)
#Note2: Very important that this list is in the same order as for the earlier datasets as sklearn ML models do not change order!!
cols = ['Delta_TP9','Delta_Fp1','Delta_Fp2','Delta_TP10',
    'Theta_TP9','Theta_Fp1','Theta_Fp2','Theta_TP10',
    'Alpha_TP9','Alpha_Fp1','Alpha_Fp2','Alpha_TP10',
    'Beta_TP9','Beta_Fp1','Beta_Fp2','Beta_TP10',
    'Gamma_TP9','Gamma_Fp1','Gamma_Fp2','Gamma_TP10']
Vals = defaultdict(list, { k:[] for k in cols})

#call class instances:
OutlierDistr = DistributionBasedOutlierDetection()
PCA = PrincipalComponentAnalysis()
ICA = IndependentComponentAnalysis()
NumAbs = NumericalAbstraction()
FreqAbs = FourierTransformation()

datapoints = 0
place = 0

rf_model = pickle.load(open("final_random_forest_model_BCI.sav", 'rb'))

def hsi_handler(address: str,*args):
    global hsi, hsi_string
    hsi = args
    if ((args[0]+args[1]+args[2]+args[3])==4):
        hsi_string_new = "Muse Fit Good"
    else:
        hsi_string_new = "Muse Fit Bad on: "
        if args[0]!=1:
            hsi_string_new += "Left Ear. "
        if args[1]!=1:
            hsi_string_new += "Left Forehead. "
        if args[2]!=1:
            hsi_string_new += "Right Forehead. "
        if args[3]!=1:
            hsi_string_new += "Right Ear."        
    if hsi_string!=hsi_string_new:
        hsi_string = hsi_string_new
        print(hsi_string)  

def wave_handler(address: str,*args):
    global Vals, datapoints, rf_model, cols
    wave = args[0][0]

	# channel configuration = [TP9, Fp1, Fp2, TP10] as per 
    # https://web.archive.org/web/20181105231756/http://developer.choosemuse.com/tools/available-data#Absolute_Band_Powers
    channels = ['TP9', 'Fp1', 'Fp2', 'TP10']

    for i in [0,1,2,3]: # for each of the 4 sensors update the specific brain wave data (delta, theta etc)
        key = wave + '_' + channels[i]
        Vals[key].append(args[i+1]) #add values to dict
        datapoints +=1

    # we have 20 features, and we want to have 3 seconds of data, data comes in 10Hz, so we first have to add 20x30=600 datapoints before moving on
    if datapoints == 600:
        # step 1: create dataframe
        # we add datetime to the df as this makes it compatible with our earlier code
        df= pd.DataFrame.from_dict(Vals,orient='index').transpose()
        df.index = pd.date_range("20180101", periods=df.shape[0], freq='100ms')

        # step 2: outlier detection
        for col in [c for c in df.columns]: 
            df = OutlierDistr.mixture_model(df, col, 3)              
            df.loc[df[f'{col}_mixture'] < 0.0005, col] = np.nan
            del df[col + '_mixture']
            df[col] = df[col].interpolate() 
            df[col] = df[col].fillna(method='bfill')

        # step 3: pre-process
        n_pcs = 4
        df = PCA.apply_pca(copy.deepcopy(df), cols, n_pcs)
        df = ICA.apply_ica(copy.deepcopy(df), cols) 

        window_sizes = [10,20,30]
        fs = 100
    
        for ws in window_sizes:          
            df = NumAbs.abstract_numerical(df, cols, ws, 
            ['mean', 'std', 'max', 'min', 'median', 'slope'])   
        df = FreqAbs.abstract_frequency(df, cols, window_sizes[0], fs)

        # now we have exactly 1 row which has no NaN, so choose that row
        df.dropna(axis=0, inplace=True)

        input = df.to_numpy()

        # step 4: predict
        pred = rf_model.predict(input)
        proba = rf_model.predict_proba(input)
        print(pred)
        print(proba)

        # step 5: update graph     
        plot_update(pred)

        #step 6: now, we reinit datapoints and the Vals dict and start again
        datapoints = 0
        Vals = defaultdict(list, { k:[] for k in cols})


def init_plot():
    ani = FuncAnimation(plt.gcf(), plot_update, interval=100) #update every 1 sec
    plt.show()


def plot_update(prediction):
    global place

    plt.cla()

    if prediction == 'label_left':
        place -= 1
    if prediction == 'label_right':
        place += 1 
    plt.plot(place,0,'ro')
    plt.xlim([-10,10])
    plt.xticks(np.arange(-10,10,1))
    plt.yticks([])


if __name__ == "__main__":
    ip = "0.0.0.0"
    port = 5000
    
    thread = threading.Thread(target=init_plot)
    thread.daemon = True
    thread.start()

    #Init Muse Listeners    
    dispatcher = dispatcher.Dispatcher()
    dispatcher.map("/muse/elements/horseshoe", hsi_handler)
    
    dispatcher.map("/muse/elements/delta_absolute", wave_handler,'Delta')
    dispatcher.map("/muse/elements/theta_absolute", wave_handler,'Theta')
    dispatcher.map("/muse/elements/alpha_absolute", wave_handler,'Alpha')
    dispatcher.map("/muse/elements/beta_absolute", wave_handler,'Beta')
    dispatcher.map("/muse/elements/gamma_absolute", wave_handler,'Gamma')

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Listening on UDP port "+str(port))
    server.serve_forever()