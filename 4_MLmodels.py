import os
import pandas as pd
from pathlib import Path
from Step4_MLmodels.PrepareDatasetForLearning import PrepareDatasetForLearning
from Step4_MLmodels.LearningAlgorithms import ClassificationAlgorithms
from Step4_MLmodels.Evaluation import ClassificationEvaluation
from Step4_MLmodels.FeatureSelection import FeatureSelectionClassification
from util import util
from util.VisualizeDataset import VisualizeDataset

# Set up file names and locations.
FOLDER_PATH = Path('./intermediate_datafiles/mental_states/step3_result')
RESULT_PATH = Path('./intermediate_datafiles/mental_states/step4_result')


def main():
    # for this script, we want to first load in all datasets
    # since the Prepare dataset function accepts a list of pd dataframes
    prepare = PrepareDatasetForLearning()

    all_datasets = []
    for condition in os.scandir(FOLDER_PATH): # go through all conditions for experiments    
        if condition.is_dir():
            condition_path = condition.path
            result_condition_path = Path(str(RESULT_PATH) +'/' +  condition.name)
            result_condition_path.mkdir(exist_ok=True, parents=True)

            for instance in os.scandir(condition_path): #instance = 1 individual experiment
                instance_path = instance.path
                dataset = pd.read_csv(instance_path, index_col=0)
                dataset.index = pd.to_datetime(dataset.index)
                all_datasets.append(dataset)

    #now all dataframes are added to the list all_datasets
    #print(all_datasets)

    # Let's create our visualization class again.
    DataViz = VisualizeDataset(__file__)

    '''
    the classification of the motor imagery can be seen as a non-temporal task.
    We first create 1 column representing our classes, and then create a train val test split of 60 20 20
    In order to do this, we first create a train test split of 80 20, and then for the train set we split again in 75 25
    For each dataset instance. we split trainvaltest split individually.
    Then later we add all train data together, all val data together, and all test data together.
    This way we sample randomly across all users to get a result for the whole 'population' of subjects.
    '''

    train_X, val_X, test_X, train_y, val_y, test_y = prepare.split_multiple_datasets_classification(
        all_datasets, ['left', 'right'], 'like', [0.8, 0.25],filter=True, temporal=False)
    print('Training set length is: ', len(train_X.index))
    print('Validation set length is: ', len(val_X.index))
    print('Test set length is: ', len(test_X.index))   

    # select subsets of features which we will consider:
    basic_features = ['Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10','Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10','Alpha_TP9','Alpha_AF7',
    'Alpha_AF8','Alpha_TP10','Beta_TP9','Beta_AF7','Beta_AF8','Beta_TP10','Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10']
    pca_features = ['pca_1','pca_2','pca_3','pca_4']
    ica_features = ['FastICA_1','FastICA_2','FastICA_3','FastICA_4','FastICA_5','FastICA_6','FastICA_7','FastICA_8','FastICA_9','FastICA_10',
    'FastICA_11','FastICA_12','FastICA_13','FastICA_14','FastICA_15','FastICA_16','FastICA_17','FastICA_18','FastICA_19','FastICA_20']
    time_features = [name for name in dataset.columns if '_temp_' in name]
    freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]

    basic_w_PCA = list(set().union(basic_features, pca_features))
    basic_w_ICA = list(set().union(basic_features, ica_features))
    all_features = list(set().union(basic_features, ica_features, time_features, freq_features))

    fs = FeatureSelectionClassification()
    num_features = 50
    # we will select the top 50 features based on an experiment with a deciscion tree
    selected_features, ordered_features, ordered_scores = fs.forward_selection(num_features,
                                                                  train_X[all_features],
                                                                  test_X[all_features],
                                                                  train_y,
                                                                  test_y,
                                                                  gridsearch=False)
         
    # then here, we run each model
    #TODO

    # and then we chose the 2 best ones to apply gridsearch etc
    #TODO

    # eventually we end up with the best one, and we save it
    #TODO


if __name__ == '__main__':
    main()