# Add window frequency and time based features
# Add ICA features
# maybe check cluster feature
# check features proposed here: https://github.com/sari-saba-sadiya/EEGExtract

import argparse
import copy
from pathlib import Path
import os
import pandas as pd
from Step_3_feature_engineering.Dim_reduction import PrincipalComponentAnalysis, IndependentComponentAnalysis
from Step_3_feature_engineering.TemporalAbstraction import NumericalAbstraction
from Step_3_feature_engineering.FrequencyAbstraction import FourierTransformation
from util.VisualizeDataset import VisualizeDataset

# Set up file names and locations.
FOLDER_PATH = Path('./intermediate_datafiles/mental_states/step2_result')
RESULT_PATH = Path('./intermediate_datafiles/mental_states/step3_result')

def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))

def main():
    print_flags()

    # We'll create an instance of our visualization class to plot results.
    DataViz = VisualizeDataset(__file__)

    # initialize feature engineering classes
    PCA = PrincipalComponentAnalysis()
    ICA = IndependentComponentAnalysis()
    NumAbs = NumericalAbstraction()
    FreqAbs = FourierTransformation()

    # initialize example dataset stuff for experimenting
    example_filename = "./intermediate_datafiles/mental_states/step2_result/Neutral/Neutral_2021-07-06--17-26-04_3567260142020984011.csv"
    example_dataset = pd.read_csv(example_filename, index_col=0)
    example_dataset.index = pd.to_datetime(example_dataset.index)

    # ms per instance is used for the freq and time features
    milliseconds_per_instance = (example_dataset.index[1] - example_dataset.index[0]).microseconds/1000

    if FLAGS.mode == 'PCA':
        selected_cols = [c for c in example_dataset.columns[1:] if not 'left' in c or 'right' in c] #select brainwave data
        pc_values = PCA.determine_pc_explained_variance(example_dataset, selected_cols)

        # Plot the variance explained.
        DataViz.plot_xy(x=[range(1, len(selected_cols)+1)], y=[pc_values],
                        xlabel='principal component number', ylabel='explained variance',
                        ylim=[0, 1], line_styles=['b-'], algo='PCA')

        # We select 4 as the best number of PC's as this explains most of the variance
        n_pcs = 4
        example_dataset = PCA.apply_pca(copy.deepcopy(example_dataset), selected_cols, n_pcs)

        # And we visualize the result of the PC's
        DataViz.plot_dataset(example_dataset, ['pca_'], ['like'], ['line'], algo='PCA')
  
    elif FLAGS.mode == 'ICA':
        selected_cols = [c for c in example_dataset.columns[1:] if not 'left' in c or 'right' in c] #select brainwave data
        example_dataset = ICA.apply_ica(copy.deepcopy(example_dataset), selected_cols) #we apply FastICA for all components (all cols)
        # And we visualize the result of the IC's
        DataViz.plot_dataset(example_dataset, ['FastICA_'], ['like'], ['line'], algo='ICA')

    elif FLAGS.mode == 'aggregation':
        # Set the window sizes to the number of instances representing 1 2, and 5seconds
        window_sizes = [int(float(1000)/milliseconds_per_instance), int(float(2000)/milliseconds_per_instance), 
        int(float(5000)/milliseconds_per_instance)]    

        for ws in window_sizes:          
            example_dataset = NumAbs.abstract_numerical(example_dataset, ['Delta_TP9'], ws, 
            ['mean', 'std', 'max', 'min', 'median', 'slope'])

        DataViz.plot_dataset(example_dataset, ['Delta_TP9', 'Delta_TP9_temp_mean', 'Delta_TP9_temp_std', 'Delta_TP9_temp_slope'], 
        ['exact', 'like', 'like', 'like'], ['line', 'line', 'line', 'line'],algo='aggregation')

    elif FLAGS.mode == 'frequency':
        # Now we move to the frequency domain, with the same window sizes.             
        fs = 100 #sample frequency

        window_sizes = [int(float(1000)/milliseconds_per_instance), int(float(2000)/milliseconds_per_instance), 
        int(float(5000)/milliseconds_per_instance)]    

        for ws in window_sizes:   
            example_dataset = FreqAbs.abstract_frequency(example_dataset, ['Delta_TP9'], ws, fs)

        DataViz.plot_dataset(example_dataset, ['Delta_TP9_max_freq', 'Delta_TP9_freq_weighted', 'Delta_TP9_pse'],
         ['like', 'like', 'like'], ['line', 'line', 'line'], algo='frequency')

    
    elif FLAGS.mode == 'final': #in final, we run the pipeline for all files
        for condition in os.scandir(FOLDER_PATH): # go through all conditions for experiments    
            if condition.is_dir():
                condition_path = condition.path
                result_condition_path = Path(str(RESULT_PATH) +'/' +  condition.name)
                result_condition_path.mkdir(exist_ok=True, parents=True)

                for instance in os.scandir(condition_path): #instance = 1 individual experiment
                    instance_path = instance.path
                    print(f'Going through pipeline for file {instance_path}.')
                    dataset = pd.read_csv(instance_path, index_col=0)
                    dataset.index = pd.to_datetime(dataset.index)
                    selected_cols = [c for c in dataset.columns if not 'left' in c or 'right' in c]

                    #PCA with n_pcs of 4 as found in experiment above
                    n_pcs = 4
                    dataset = PCA.apply_pca(copy.deepcopy(dataset), selected_cols, n_pcs)

                    #DataViz.plot_dataset(dataset, ['Delta_TP9', 'Theta_AF7', 'Alpha_AF8', 'Beta_TP10', 'Gamma_AF7', 'pca_1'],
                    # ['like', 'like', 'like', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'line', 'line', 'line'], algo='finalPCA')
                    
                    #ICA: we apply FastICA for all components (all cols)
                    dataset = ICA.apply_ica(copy.deepcopy(dataset), selected_cols) 

                    #DataViz.plot_dataset(dataset, ['Delta_TP9', 'Theta_AF7', 'Alpha_AF8', 'Beta_TP10', 'Gamma_AF7', 'FastICA_1'],
                    # ['like', 'like', 'like', 'like', 'like', 'like', 'like'], ['line', 'line', 'line', 'line', 'line', 'line', 'line'], algo='finalICA')

                    # Freq and time domain features for ws of 1 sec, 2 sec, and 3 sec
                    window_sizes = [int(float(1000)/milliseconds_per_instance), int(float(2000)/milliseconds_per_instance),
                    int(float(3000)/milliseconds_per_instance)]
                    fs = 100 #sample frequency
               
                    for ws in window_sizes:          
                        dataset = NumAbs.abstract_numerical(dataset, selected_cols, ws, 
                        ['mean', 'std', 'max', 'min', 'median', 'slope'])   
                    
                    # we only do fourier transformation for smallest ws [1 sec]
                    dataset = FreqAbs.abstract_frequency(dataset, selected_cols, window_sizes[0], fs)

                    # Now we only take a certain percentage of overlap in the windows, otherwise our training examples will be too much alike.
                    # The percentage of overlap we allow:
                    window_overlap = 0.5
                    # we do this for the biggest ws
                    skip_points = int((1-window_overlap) * window_sizes[-1])
                    dataset = dataset.iloc[::skip_points,:]

                    #apparently the first two rows are NaNs so delete those.
                    dataset = dataset.iloc[2:]

                    DataViz.plot_dataset(dataset, ['Delta_TP9', 'Theta_AF7', 'Alpha_AF8', 'Beta_TP10', 'Gamma_AF7', 'pca_1', 'FastICA_1'],
                     ['like', 'like', 'like', 'like', 'like', 'like', 'like', 'like'], 
                     ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'line'], algo='final_step3')

                    # save data
                    dataset.to_csv(Path(str(result_condition_path) + '/' + instance.name))
                    #print(dataset.shape)


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: final, ICA, PCA, aggregation, frequency \
                        'ICA' is to study the effect of ICA and plot the results \
                        'PCA' is to study the effect of PCA and plot the results\
                        'aggregation' is to study effects of temporal aggregation methods \
                        'frequency' is to study effects of fast Fourier transformation methods \
                        'final' is used for the next step", choices=['ICA', 'PCA', 'aggregation', 'frequency', 'final'])

    FLAGS, unparsed = parser.parse_known_args()
    main()