'''
ch2
1) Set on granularity? Not possible for the raw signals at least since that freq is just barely high enough
2) If needed, change time to UNIX timestamp
3) Visualize dataset

ch3
1) notch filter for 50Hz?
2) detect outliers and zero-values??
3) add PCA

ch4
1) add temporal and frequency features for each window and each sensor:
    - Mean 
    - Std
    - Max
    - Min
    - Slope
    - FFT Max freq
    - FFT Weighted signal average
    - FFT Power spectral entropy
Note that these are 8 features, for 5 sensors (of which 1 is the ground electrode so should we use that?).
Thus resulting in 8x5 = 40 features per window.

ch7
1) classification
'''