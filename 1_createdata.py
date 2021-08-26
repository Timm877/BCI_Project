from Step_1_create_dataset.CreateDataset import CreateDataset
from util.VisualizeDataset import VisualizeDataset
from util import util
from pathlib import Path
import copy
import os
import sys

# directory paths for the current experiment (change it if needed)
FOLDER_PATH = Path('./datasets/mental_states/')
RESULT_PATH = Path('./intermediate_datafiles/mental_states/step1_result/')
RESULT_PATH.mkdir(exist_ok=True, parents=True)
GRANULARITY = 100 #milisecond per instance; we settle at 100 ms.


for condition in os.scandir(FOLDER_PATH): # go through all conditions for experiments    
    if condition.is_dir():
        condition_path = condition.path
        result_condition_path = Path(str(RESULT_PATH) + condition.name)
        result_condition_path.mkdir(exist_ok=True, parents=True)

        for instance in os.scandir(condition_path): 
            # an instance = 1 individual experiment
            # When doing the left right experiment, we have 15 files per subject
            # train/val/test split would be 60/20/20 which we want to pick out randomly
            # we dont concat the data now as that creates weird bumps between experiments
            # but after all pre-processing is done, we create trainvaltest splits for each dataset
            # and add all train data together for training etc
            instance_path = instance.path
            print(f'Creating numerical datasets for {instance_path} using granularity {GRANULARITY}.')
            dataset = CreateDataset(instance_path, GRANULARITY)

            # We add the brain wave data, for each sensor, for each brain wave
            # and aggregate the values per timestep by averaging the values
            # TODO currently, nothing is done with left right labels yet.
            dataset = dataset.add_data(instance_path, ['Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10',
            'Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10',
            'Alpha_TP9','Alpha_AF7','Alpha_AF8','Alpha_TP10',
            'Beta_TP9','Beta_AF7','Beta_AF8','Beta_TP10',
            'Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10'], ['left', 'right'], 'avg')

            # Plot the data
            DataViz = VisualizeDataset(__file__)
            # 1. Boxplot
            DataViz.plot_dataset_boxplot(dataset, ['Delta_TP9','Theta_TP9', 'Alpha_TP9', 'Beta_TP9', 'Gamma_TP9',
            'Delta_AF8','Theta_AF8','Alpha_AF8','Beta_AF8','Gamma_AF8'], instance.name.split('--')[0])
            # 2. Plot brainwaves: we plot as per example the waves for af8
            DataViz.plot_dataset(dataset, ['Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10',
            'Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10'],
                                            ['like', 'like', 'like', 'like', 'like', 'like', 'like','like'],
                                            ['line', 'line', 'line', 'line', 'line', 'line', 'line', 'line'], instance.name.split('--')[0])
            # 3. And we print a summary of the dataset.
            util.print_statistics(dataset)

            # Store the dataset we generated.
            dataset.to_csv(Path(str(result_condition_path) + '/' + instance.name))