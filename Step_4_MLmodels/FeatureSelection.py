from Step_4_MLmodels.LearningAlgorithms import ClassificationAlgorithms
from Step_4_MLmodels.Evaluation import ClassificationEvaluation
from scipy.stats import pearsonr
import sys
import copy
import numpy as np
from operator import itemgetter

# Specifies feature selection approaches for classification to identify the most important features.
class FeatureSelectionClassification:
    
    # Forward selection for classification which selects a pre-defined number of features (max_features)
    # that show the best accuracy. We assume a decision tree learning for this purpose, but
    # this can easily be changed. It return the best features.
    def forward_selection(self, max_features, X_train, X_test, y_train, y_test, gridsearch):
        # Start with no features.
        ordered_features = []
        ordered_scores = []
        selected_features = []
        ca = ClassificationAlgorithms()
        ce = ClassificationEvaluation()
        prev_best_perf = 0

        # Select the appropriate number of features.
        for i in range(0, max_features):
            # Determine the features left to select.
            features_left = list(set(X_train.columns) - set(selected_features))
            best_perf = 0
            best_attribute = ''

            print("Added feature{}".format(i))
            # For all features we can still select...
            for f in features_left:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.append(f)

                # Determine the accuracy of a decision tree learner if we were to add
                # the feature.
                pred_y_train, pred_y_test, prob_training_y, prob_test_y = ca.decision_tree(X_train[temp_selected_features],
                                                                                           y_train,
                                                                                           X_test[temp_selected_features],
                                                                                           gridsearch=False)
                perf = ce.accuracy(y_test, pred_y_test)

                # If the performance is better than what we have seen so far (we aim for high accuracy)
                # we set the current feature to the best feature and the same for the best performance.
                if perf > best_perf:
                    best_perf = perf
                    best_feature = f

            # We select the feature with the best performance.
            selected_features.append(best_feature)
            prev_best_perf = best_perf
            ordered_features.append(best_feature)
            ordered_scores.append(best_perf)

        return selected_features, ordered_features, ordered_scores

    # Backward selection for classification which selects a pre-defined number of features (max_features)
    # that show the best accuracy. We assume a decision tree learning for this purpose, but
    # this can easily be changed. It return the best features.
    def backward_selection(self, max_features, X_train, y_train):
        # First select all features.
        selected_features = X_train.columns.tolist()
        ca = ClassificationAlgorithms()
        ce = ClassificationEvaluation()
        for i in range(0, (len(X_train.columns) - max_features)):
            best_perf = 0
            worst_feature = ''

            # Select from the features that are still in the selection.
            for f in selected_features:
                temp_selected_features = copy.deepcopy(selected_features)
                temp_selected_features.remove(f)

                # Determine the score without the feature.
                pred_y_train, pred_y_test, prob_training_y, prob_test_y = ca.decision_tree(X_train[temp_selected_features], y_train, X_train[temp_selected_features])
                perf = ce.accuracy(y_train, pred_y_train)

                # If we score better without the feature than what we have seen so far
                # this is the worst feature.
                if perf > best_perf:
                    best_perf = perf
                    worst_feature = f

            # Remove the worst feature.
            selected_features.remove(worst_feature)
        return selected_features