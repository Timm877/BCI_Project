# BCI_Project: Classification of mental states

In this project, it is attempted to read collect data from brain signals and from these signals, predict the mental state. 

## Part 1: Theoreticals and starting up
**Introduction:** Brain-computer interfaces exist for a couple of decades now. Only recently however, due to the increase in compute, advancements in ML and DL, and advancements in hardware, significant steps are made to improve the decoding of brain signals for useful applications. Various measurement methods exist. These can be classified as being invasive and non-invasive. Invasive means that electrodes are placed directly into the skull of the subject (iEEG, sEEG, Neuralink, infrared). Non-invasive (EEG) can simply be put over the skull, making it less complex to apply and less impactful for the patient. However, the signal-to-noise ratio is higher for the EEG measurements, making it less accurate. However, a lot of effort is done in making the EEG hardware easier to wear, and cheaper to buy, making it also a hot topic to research to improve predictions based on EEG data, as a accurate system is widely apllicable.

**EEG measurements with Muse:** Recent years, advancements are made in controlling an exoskeleton using a BCI **ref**, to predict mental states**ref**, and learning more about the functions of the brain in general**ref**. Most of these studies use rather an extensice, 32-channel system which is quite expenisive. Fortunately, also cheaper options are on the market, such as the Muse 2, which is considered to be a low cost device with good accuracy, suitable for our project! The Muse headset has 5 dry electrodes. Dry electrodes means that ... Also wet electrodes.. placed at TP9 and TP10, AF7 and AF8, and NZ (reference electrode) (see image). Explain the 1-20 system. 

**Basic neuroscience:** Neurons in the brain ... Action potential ... Brain areas and functions (image)  ... Coming back to our Muse defin, we discussed that the placement of the electrodes are at TP (temporoparital?) and Af (frontal). Accosiated with ..
It has been found out that neurons that wire together, fire together. A group of neurons firing together results in a higher amplitude in our EEG signals. It was observed that large groups of neurons tend to fire together once in a while, while smaller groups more occionasely fire together. Later, these signals were combined with the brain function at that moment. It was observed that for certain tasks and mental states, the amplitude?? or the aanwezigheid?? waves in specific frequency intervals increased. This resulted in the following distinction (see image): 
- $\gamma$, 32-100Hz with lowest amplitude. Associated with heightened perception, learning, problem solving.
- $\beta$, 13-32 Hz. Associated with awakeness, thinking, excitement
- $\alpha$, 8-13 Hz. Associated with physically and mentally relaxed state.
- $\theta$, 4-8 Hz. Associated with creativity, insight, deep meditation.
- $\delta$, 0.5-4 Hz, with highest amplitude. Deep sleep.

Based on the description of the brain waves, one would suspect that classifying mental tasks would be possible. For the neutral tasks, the amplitude for the $\beta$ waves would be higher, as the state is just normal awakeness and thinking. For the meditative state, the $\alpha$ waves would rise. Also $\theta$ waves could rise due to the association with deep meditation, but since the task is only for 1 minute, this timeframe seems to be too low to reach a deep meditative state, especially for the participants who are untrained in meditation. Lastly, we have the concentrated state, for which it is expected that the $\gamma$ waves could rise.

This made me decide to focus on this specific task: classifying mental states from Muse 2 EEG headset data using machine learning.

## Part 2: Experimental design and collecting data!
I use the Mind-monitor app, made by for putting Muse data to computer. This app is pretty awesome. In the app, you're able to measure your brain waves real-time. Or look at a real-time FFT, or spectogram (show image). Make sure to adjust the notch filter for your net Hz (I live in Europe, so i changed it to 50Hz).The creator of this app also has some starter code available on Github, to receive the streaming data, and saving it to a CSV or creating a live animation in Matplotlib. In the app, change the IP-adress to your WiFI IP adress, copy the code from the repo, and try it out!

1. **Collecting data:** 
    - After a quick google search, I found an article with the following setup: EEG data from different participants is collected using the Muse 2 EEG Headset, in a Meditative, Concentrated and Neutral mental state, 1 minute each. The states are defined as: 
        - Neutral: just relaxing, without any stimuli.
        - Meditative: relaxing, while listining to meditation music
        - Concentrated: trying to follow a game of "balletje-balletje (shell game)" 
    This sounds like a perfect setup to use, and to compare our results with theirs!
2. The features we currently have in our dataset are:
    - Raw signals for TP9 and TP10 , AF7 and AF8, and NZ (reference electrode)
    - The brain wave signal averaged over all electrodes, processed by the Mind-monitor app. Consists of: $\gamma$, $\beta$, $\alpha$, $\theta$, $\delta$.
    
## Part 3: Pre-processing and feature engineering!  
part1: granularity
part2: outlier detection
part3: filter and PCA
part4: add temporal and frequency features for each window and each sensor:
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

## Part 4: Training classifiers!  

## Part 5: Real-time predictions! 