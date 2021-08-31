import os
import pandas as pd
from pathlib import Path
from Step_4_MLmodels.PrepareDatasetForLearning import PrepareDatasetForLearning
from Step_4_MLmodels.LearningAlgorithms import ClassificationAlgorithms
from Step_4_MLmodels.Evaluation import ClassificationEvaluation
from Step_4_MLmodels.FeatureSelection import FeatureSelectionClassification
from util import util
from util.VisualizeDataset import VisualizeDataset

# Set up file names and locations.
FOLDER_PATH = Path('./intermediate_datafiles/motor_imagery/step3_result')
RESULT_PATH = Path('./intermediate_datafiles/motor_imagery/step4_result')


def main():

    RESULT_PATH.mkdir(exist_ok=True, parents=True)
    # for this script, we want to first load in all datasets
    # since the Prepare dataset function accepts a list of pd dataframes
    prepare = PrepareDatasetForLearning()
    all_datasets = []

    for instance in os.scandir(FOLDER_PATH): # go through all instances of experiments
        instance_path = instance.path
        dataset = pd.read_csv(instance_path, index_col=0)
        dataset.index = pd.to_datetime(dataset.index)
        all_datasets.append(dataset)

    #now all dataframes are added to the list all_datasets
    #print(all_datasets)

    # Let's create our visualization class again.
    DataViz = VisualizeDataset(__file__)

    '''
    the classification of the motor imagery can be seen as a non-temporal task, as we want to predict imagery based on a window of e.g. 2 sec,
    without taking into account previous windows.
    We first create 1 column representing our classes, and then create a train val test split of 60 20 20
    In order to do this, we first create a train test split of 80 20, and then for the train set we split again in 75 25
    For each dataset instance. we split trainvaltest split individually.
    Then later we add all train data together, all val data together, and all test data together.
    This way we sample randomly across all users to get a result for the whole 'population' of subjects.
    '''
    # we set filter is false so also the data besides left and right are taken with us
    train_X, val_X, test_X, train_y, val_y, test_y = prepare.split_multiple_datasets_classification(
        all_datasets, ['label_left', 'label_right'], 'like', [0.2, 0.25],filter=False, temporal=False)
    print('Training set length is: ', len(train_X.index))
    print('Validation set length is: ', len(val_X.index))
    print('Test set length is: ', len(test_X.index))   

    # select subsets of features which we will consider:
    pca_features = ['pca_1','pca_2','pca_3','pca_4']
    ica_features = ['FastICA_1','FastICA_2','FastICA_3','FastICA_4','FastICA_5','FastICA_6','FastICA_7','FastICA_8','FastICA_9','FastICA_10',
    'FastICA_11','FastICA_12','FastICA_13','FastICA_14','FastICA_15','FastICA_16','FastICA_17','FastICA_18','FastICA_19','FastICA_20']
    time_features = [name for name in dataset.columns if '_temp_' in name]
    freq_features = [name for name in dataset.columns if (('_freq' in name) or ('_pse' in name))]


    # feature selection below we will use as input for our models:
    basic_features = ['Delta_TP9','Delta_AF7','Delta_AF8','Delta_TP10','Theta_TP9','Theta_AF7','Theta_AF8','Theta_TP10','Alpha_TP9','Alpha_AF7',
    'Alpha_AF8','Alpha_TP10','Beta_TP9','Beta_AF7','Beta_AF8','Beta_TP10','Gamma_TP9','Gamma_AF7','Gamma_AF8','Gamma_TP10']
    basic_w_PCA = list(set().union(basic_features, pca_features))
    basic_w_ICA = list(set().union(basic_features, ica_features))
    all_features = list(set().union(basic_features, ica_features, time_features, freq_features))

    fs = FeatureSelectionClassification()
    num_features = 20

    # we will select the top 20 features based on an experiment with a deciscion tree which we will use as input for our models as well
    # this is already been run, see below

    '''
    selected_features, ordered_features, ordered_scores = fs.forward_selection(num_features,
                                                                  train_X[all_features],
                                                                  test_X[all_features],
                                                                  train_y,
                                                                  test_y,
                                                                  gridsearch=False)
    print(selected_features)
    '''

    # the best feature are right now:
    selected_features = ['Delta_AF7_temp_max_ws_10', 'Alpha_TP9_temp_mean_ws_10', 'Delta_AF7_temp_slope_ws_30', 'FastICA_2', 
    'Alpha_TP9_temp_median_ws_20', 'Delta_AF8_temp_max_ws_10', 'Beta_TP10_freq_30.0_Hz_ws_10', 
    'Beta_TP9_temp_std_ws_20', 'Theta_TP10_temp_max_ws_20', 'Gamma_TP9_temp_median_ws_20',
     'Gamma_TP10_freq_30.0_Hz_ws_10', 'Alpha_TP10_temp_std_ws_20', 'Gamma_AF7_freq_30.0_Hz_ws_10', 'Delta_TP10', 
     'Beta_TP9_temp_median_ws_20', 'Delta_TP10_temp_min_ws_20', 'Theta_TP9_temp_median_ws_30', 
     'Delta_AF8_temp_min_ws_20', 'Delta_AF8_temp_mean_ws_10', 'Beta_TP9_freq_0.0_Hz_ws_10']


    possible_feature_sets = [basic_features, basic_w_PCA, basic_w_ICA, all_features, selected_features]
    feature_names = ['initial set', 'basic_w_PCA', 'basic_w_ICA', 'all_features', 'Selected features']
    N_KCV_REPEATS = 10 # some non deterministic models we will run a couple of times as their inits are random to get average results


    # then here, we run each model
    learner = ClassificationAlgorithms()
    eval = ClassificationEvaluation()
    scores_over_all_algs = []
    '''
    for i in range(0, len(possible_feature_sets)):

        selected_train_X = train_X[possible_feature_sets[i]]
        selected_val_X = val_X[possible_feature_sets[i]]

        performance_training_nn = 0
        performance_training_rf = 0
        performance_training_svm = 0
        performance_validation_nn = 0
        performance_validation_rf = 0
        performance_validation_svm = 0

        # first the non deterministic models for which we average over 5 runs
        for repeat in range(0, N_KCV_REPEATS):
            print("Training NeuralNetwork run {} / {} ... ".format(repeat+1, N_KCV_REPEATS, feature_names[i]))
            class_train_y, class_val_y, class_train_prob_y, class_val_prob_y = learner.feedforward_neural_network(
                selected_train_X, train_y, selected_val_X, gridsearch=False
            )
            performance_training_nn += eval.f1(train_y, class_train_y)
            performance_validation_nn += eval.f1(val_y, class_val_y)


            print("Training RandomForest run {} / {} ... ".format(repeat+1, N_KCV_REPEATS, feature_names[i]))
            class_train_y, class_val_y, class_train_prob_y, class_val_prob_y = learner.random_forest(
                selected_train_X, train_y, selected_val_X, gridsearch=False
            )
            performance_training_rf += eval.f1(train_y, class_train_y)
            performance_validation_rf += eval.f1(val_y, class_val_y)


            print("Training SVM run {} / {}, featureset: {}... ".format(repeat+1, N_KCV_REPEATS, feature_names[i]))
            class_train_y, class_val_y, class_train_prob_y, class_val_prob_y = learner.support_vector_machine_with_kernel(
                selected_train_X, train_y, selected_val_X, gridsearch=False
            )
            performance_training_svm += eval.f1(train_y, class_train_y)
            performance_validation_svm += eval.f1(val_y, class_val_y)


        overall_performance_training_nn = performance_training_nn/N_KCV_REPEATS
        overall_performance_validation_nn = performance_validation_nn/N_KCV_REPEATS
        overall_performance_training_rf = performance_training_rf/N_KCV_REPEATS
        overall_performance_validation_rf = performance_validation_rf/N_KCV_REPEATS
        overall_performance_training_svm = performance_training_svm/N_KCV_REPEATS
        overall_performance_validation_svm = performance_validation_svm/N_KCV_REPEATS

        # And we run our deterministic classifiers:
        print("Determenistic Classifiers:")
        print("Training Nearest Neighbor run 1 / 1, featureset {}:".format(feature_names[i]))
        class_train_y, class_val_y, class_train_prob_y, class_val_prob_y = learner.k_nearest_neighbor(
            selected_train_X, train_y, selected_val_X, gridsearch=False
        )
        performance_training_knn = eval.f1(train_y, class_train_y)
        performance_validation_knn = eval.f1(val_y, class_val_y)


        print("Training Descision Tree run 1 / 1  featureset {}:".format(feature_names[i]))
        class_train_y, class_val_y, class_train_prob_y, class_val_prob_y = learner.decision_tree(
            selected_train_X, train_y, selected_val_X, gridsearch=False
        )
        performance_training_dt = eval.f1(train_y, class_train_y)
        performance_validation_dt = eval.f1(val_y, class_val_y)


        print("Training Naive Bayes run 1/1 featureset {}:".format(feature_names[i]))
        class_train_y, class_val_y, class_train_prob_y, class_val_prob_y = learner.naive_bayes(
            selected_train_X, train_y, selected_val_X
        )
        performance_training_nb = eval.f1(train_y, class_train_y)
        performance_validation_nb = eval.f1(val_y, class_val_y)

        print("Training LDA run 1/1 featureset {}:".format(feature_names[i]))
        class_train_y, class_val_y, class_train_prob_y, class_val_prob_y = learner.LDA(
            selected_train_X, train_y, selected_val_X
        )
        performance_training_LDA = eval.f1(train_y, class_train_y)
        performance_validation_LDA = eval.f1(val_y, class_val_y)
        print (performance_training_LDA)

        scores_with_sd = util.print_table_row_performances(feature_names[i], len(selected_train_X.index), len(selected_val_X.index), [
                                                                                            (overall_performance_training_nn, overall_performance_validation_nn),
                                                                                            (overall_performance_training_rf, overall_performance_validation_rf),
                                                                                            (overall_performance_training_svm, overall_performance_validation_svm),
                                                                                            (performance_training_knn, performance_validation_knn),
                                                                                            (performance_training_dt, performance_validation_dt),
                                                                                            (performance_training_nb, performance_validation_nb),
                                                                                            (performance_training_LDA, performance_validation_LDA)
                                                                                            ])
        scores_over_all_algs.append(scores_with_sd)

    DataViz.plot_performances_classification(['NN', 'RF','SVM', 'KNN', 'DT', 'NB', 'LDA'], feature_names, scores_over_all_algs)
    # we plot validation results together with their std
    '''

    # and then we chose the 1 or 2 best ones to apply gridsearch etc
    # from my initial results, RF with all features seems to perform best!
    # lets try it with the validation set and gridsearch = True.
    # eventually if we are happy with the best one, and we save that by setting save_model=True for later use with the real time predictions part!
    #print(test_y)
    #print(train_X)
    class_train_y, class_test_y, class_train_prob_y, class_test_prob_y = learner.random_forest(
                train_X, train_y, test_X, gridsearch=True, print_model_details=True, save_model=True
            )
    performance_training_rf_final = eval.f1(train_y, class_train_y)
    performance_test_rf_final = eval.f1(test_y, class_test_y)
    confusionmatrix_rf_final = eval.confusion_matrix(test_y, class_test_y, ['label_left', 'label_right', 'undefined'])
    print(performance_test_rf_final) #test performance is reasonable!
    print(confusionmatrix_rf_final)


if __name__ == '__main__':
    main()