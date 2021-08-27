from util.VisualizeDataset import VisualizeDataset
from Step_2_preprocess.OutlierDetection import DistributionBasedOutlierDetection, DistanceBasedOutlierDetection
from Step_2_preprocess.Filters import Filters
import os
import sys
import copy
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

# Set up file names and locations.
FOLDER_PATH = Path('./intermediate_datafiles/motor_imagery/step1_result')
RESULT_PATH = Path('./intermediate_datafiles/motor_imagery/step2_result')

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

    #initialize outlier classes
    OutlierDistr = DistributionBasedOutlierDetection()
    OutlierDist = DistanceBasedOutlierDetection()

    # all methods, except 'final', are for experimentation
    # for those, we choose random features for the expertiments to inspect
    outlier_columns = ['Delta_TP9', 'Beta_AF7']
    example_filename = "./intermediate_datafiles/motor_imagery/step1_result/med_2021-08-26--21-03-58_7419762883904244960.csv"
    example_dataset = pd.read_csv(example_filename, index_col=0)
    example_dataset.index = pd.to_datetime(example_dataset.index)

    if FLAGS.mode == 'chauvenet':
        for col in outlier_columns:
            print(f"Applying Chauvenet outlier criteria for column {col} with c component {FLAGS.c}")
            example_dataset = OutlierDistr.chauvenet(example_dataset, col, FLAGS.c)
            print('Number of outliers for ' + col + ': ' + str(example_dataset[col + '_outlier'].sum()))
            DataViz.plot_binary_outliers(example_dataset, col, col + '_outlier', 'chauvenet')

    elif FLAGS.mode == 'mixture':
        for col in outlier_columns:
            print(f"Applying mixture model for column {col} with n component {FLAGS.n}")
            example_dataset = OutlierDistr.mixture_model(example_dataset, col, FLAGS.n)
            print('Number of outliers for points with prob < 5e-5 for feature ' + col + ': ' + str(example_dataset[col+'_mixture'][example_dataset[col+'_mixture'] < 0.0005].count()))
            DataViz.plot_dataset(example_dataset, [col, col + '_mixture'], ['exact', 'exact'], ['line', 'points'], algo='mixture', filename='2021-08-26--18-45-49')

    elif FLAGS.mode == 'distance':
        for col in outlier_columns:
            print(f"Applying simple distance based for column {col}")
            try:
                dataset_dist = OutlierDist.simple_distance_based(example_dataset, [col], 'euclidean', FLAGS.dmin, FLAGS.fmin)
                print('Number of outliers for ' + col + ': ' + str(dataset_dist['simple_dist_outlier'].sum()))
                DataViz.plot_binary_outliers(dataset_dist, col, 'simple_dist_outlier', 'distance')
            except MemoryError as e:
                print(
                    'Not enough memory available for simple distance-based outlier detection...')
                print('Skipping.')

    elif FLAGS.mode == 'LOF':
        for col in outlier_columns:
            try:
                dataset_lof = OutlierDist.local_outlier_factor(example_dataset, [col], 'euclidean', FLAGS.K)
                print('Number of outliers for points with prob > 2 for feature ' + col + ': ' + str(dataset_lof['lof'][dataset_lof['lof'] > 2].count()))
                DataViz.plot_dataset(dataset_lof, [col, 'lof'], ['exact', 'exact'], ['line', 'points'],algo='LOF')
            except MemoryError as e:
                print('Not enough memory available for lof...')
                print('Skipping.')
          
    elif FLAGS.mode == 'final': #in final, we run the pipeline for all files
        RESULT_PATH.mkdir(exist_ok=True, parents=True)

        for instance in os.scandir(FOLDER_PATH): # go through all instances of experiments  
            instance_path = instance.path
            print(f'Going through pipeline for file {instance_path}.')
            dataset = pd.read_csv(instance_path, index_col=0)
            dataset.index = pd.to_datetime(dataset.index)

            for col in [c for c in dataset.columns if not 'label' in c]: 
                print(f'Measurement is now: {col}')
                #print('Step 1: Outlier detection')

                # we use mixture model as it is used in one paper with n=3. Number of outliers is very low 
                # but measurements are short so this is explainable, also we use brain wave data now
                # in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7728142/pdf/sensors-20-06730.pdf
                # they actually check sort of manually what data is noisy and discard that
                # I did that as well, by looking at the figure I made in step 1 for creating the dataset

                dataset = OutlierDistr.mixture_model(dataset, col, FLAGS.n)
                #print('Number of outliers for points with prob < 5e-5 for feature ' + col + ': ' + str(dataset[col+'_mixture'][dataset[col+'_mixture'] < 0.0005].count()))
                
                dataset.loc[dataset[f'{col}_mixture'] < 0.0005, col] = np.nan
                del dataset[col + '_mixture']

                #print('Step 2: Imputation')
                #print('Before interpolation, number of nans left should be > 0: ' + str(dataset[col].isna().sum()))
                #print('Also count amount of zeroes:' + str((dataset[col] == 0).sum()))

                dataset[col] = dataset[col].interpolate() #interpolating missing values
                dataset[col] = dataset[col].fillna(method='bfill') # And fill the initial data points if needed

                # check if all nan are filled in
                print('Check, number of nans left should be 0: ' + str(dataset[col].isna().sum()))




                # Step 3: lowpass filtering of periodic measurements. As all our features are brain waves and thus periodic, 
                # we do this for all features expect the labels
                # Note that the brain wave values are already filtered as per https://mind-monitor.com/Technical_Manual.php#help_graph_absolute
                # but if I later want to work with Raw EEG data, filtering is abosutely necessary
                # I would NOT use a High pass filter (https://sapienlabs.org/pitfalls-of-filtering-the-eeg-signal/)
                # which IS currently used by the mind monitor / muse app as the delta freqs are 1-4Hz
                # dataset = Filters.low_pass_filter(dataset, col, fs, cutoff, order=10)
                # dataset[col] = dataset[col + '_lowpass']
                # del dataset[col + '_lowpass']
                
            DataViz.plot_dataset(dataset, ['Gamma_','Beta_', 'Alpha_', 'Theta_', 'Delta_', 'label_'],
                    ['like', 'like', 'like', 'like', 'like', 'like'],
                    ['line', 'line', 'line', 'line', 'line', 'line'], instance.name.split('_')[1])
            # Step 4: save the file
            #print(dataset.head())
            dataset.to_csv(Path(str(RESULT_PATH) + '/' + instance.name))

if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='final',
                        help="Select what version to run: LOF, distance, mixture, chauvenet or final \
                        'LOF' applies the Local Outlier Factor to a single variable \
                        'distance' applies a distance based outlier detection method to a single variable \
                        'mixture' applies a mixture model to detect outliers for a single variable\
                        'chauvenet' applies Chauvenet outlier detection method to a single variable \
                        'final' is used for the next chapter", choices=['LOF', 'distance', 'mixture', 'chauvenet', 'final'])
    parser.add_argument('--c', type=float, default=2,
                        help="Chauvenet criterion: c component")
    parser.add_argument('--n', type=int, default=3,
                        help="Mixture model: n component")
    parser.add_argument('--K', type=int, default=5,
                        help="Local Outlier Factor:  K is the number of neighboring points considered")
    parser.add_argument('--dmin', type=float, default=0.10,
                        help="Simple distance based:  dmin is minimal distance")
    parser.add_argument('--fmin', type=float, default=0.99,
                        help="Simple distance based:  fmin is minimal fraction of points in dmin")

    FLAGS, unparsed = parser.parse_known_args()
    main()

