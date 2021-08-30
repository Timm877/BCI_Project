from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
import os
import pickle
import inspect

class ClassificationAlgorithms:

    # Apply a neural network for classification upon the training data (with the specified composition of
    # hidden layers and number of iterations), and use the created network to predict the outcome for both the
    # test and validation set. It returns the categorical predictions for the training and validation set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def feedforward_neural_network(self, train_X, train_y, test_X, save_model=False, hidden_layer_sizes=(100,), max_iter=500, activation='logistic', alpha=0.0001, learning_rate='adaptive', gridsearch=True, print_model_details=False):
        if gridsearch:
            # With the current parameters for max_iter and Python 3 packages convergence is not always reached, therefore increased +1000.
            tuned_parameters = [{'hidden_layer_sizes': [(5,), (10,), (25,), (100,), (100,5,), (100,10,),], 'activation': [activation],
                                 'learning_rate': [learning_rate], 'max_iter': [2000, 3000], 'alpha': [alpha]}]
            nn = GridSearchCV(MLPClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            # Create the model
            nn = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=activation, max_iter=max_iter, learning_rate=learning_rate, alpha=alpha, random_state=42)

        # Fit the model
        nn.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(nn.best_params_)

        if gridsearch:
            nn = nn.best_estimator_

        # Apply the model
        pred_prob_training_y = nn.predict_proba(train_X)
        pred_prob_test_y = nn.predict_proba(test_X)
        pred_training_y = nn.predict(train_X)
        pred_test_y = nn.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nn.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nn.classes_)
        if save_model:
            # save the model to disk
            filename = 'final_' + str(inspect.stack()[0][3]) + '_model_BCI.sav'
            pickle.dump(nn, open(filename, 'wb'))

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    def LDA(self, train_X, train_y, test_X, gridsearch=True, print_model_details=False, save_model=False):
        # Create the model
        lda = LinearDiscriminantAnalysis()

        # Fit the model
        lda.fit(train_X, train_y.values.ravel())

        # Apply the model
        pred_prob_training_y = lda.predict_proba(train_X)
        pred_prob_test_y = lda.predict_proba(test_X)
        pred_training_y = lda.predict(train_X)
        pred_test_y = lda.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=lda.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=lda.classes_)

        if save_model:
            # save the model to disk
            filename = 'final_' + str(inspect.stack()[0][3]) + '_model_BCI.sav'
            pickle.dump(lda, open(filename, 'wb'))

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a support vector machine for classification upon the training data (with the specified value for
    # C, epsilon and the kernel function), and use the created model to predict the outcome for both the
    # training and validation set. It returns the categorical predictions for the training and validation set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def support_vector_machine_with_kernel(self, train_X, train_y, test_X, C=1, save_model=False, kernel='rbf', gamma=1e-3, gridsearch=True, print_model_details=False):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                         'C': [1, 10, 100]}]
            svm = GridSearchCV(SVC(probability=True), tuned_parameters, cv=5, scoring='accuracy')
        else:
            svm = SVC(C=C, kernel=kernel, gamma=gamma, probability=True, cache_size=7000)

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model
        pred_prob_training_y = svm.predict_proba(train_X)
        pred_prob_test_y = svm.predict_proba(test_X)
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)

        if save_model:
            # save the model to disk
            filename = 'final_' + str(inspect.stack()[0][3]) + '_model_BCI.sav'
            pickle.dump(svm, open(filename, 'wb'))

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a support vector machine for classification upon the training data (with the specified value for
    # C, epsilon and the kernel function), and use the created model to predict the outcome for both the
    # training and validation set. It returns the categorical predictions for the training and validation set set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def support_vector_machine_without_kernel(self, train_X, train_y, test_X, save_model=False, C=1, tol=1e-3, max_iter=1000, gridsearch=True, print_model_details=False):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'max_iter': [1000, 2000], 'tol': [1e-3, 1e-4],
                         'C': [1, 10, 100]}]
            svm = GridSearchCV(LinearSVC(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            svm = LinearSVC(C=C, tol=tol, max_iter=max_iter)

        # Fit the model
        svm.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(svm.best_params_)

        if gridsearch:
            svm = svm.best_estimator_

        # Apply the model
        distance_training_platt = 1/(1+np.exp(svm.decision_function(train_X)))
        pred_prob_training_y = distance_training_platt / distance_training_platt.sum(axis=1)[:,None]
        distance_test_platt = 1/(1+np.exp(svm.decision_function(test_X)))
        pred_prob_test_y = distance_test_platt / distance_test_platt.sum(axis=1)[:,None]
        pred_training_y = svm.predict(train_X)
        pred_test_y = svm.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=svm.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=svm.classes_)
        if save_model:
            # save the model to disk
            filename = 'final_' + str(inspect.stack()[0][3]) + '_model_BCI.sav'
            pickle.dump(svm, open(filename, 'wb'))

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a nearest neighbor approach for classification upon the training data (with the specified value for
    # k), and use the created model to predict the outcome for both the
    # training and validation set. It returns the categorical predictions for the training and validation set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def k_nearest_neighbor(self, train_X, train_y, test_X, n_neighbors=5, gridsearch=True, print_model_details=False, save_model = False):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'n_neighbors': [1, 2, 5, 10]}]
            knn = GridSearchCV(KNeighborsClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)

        # Fit the model
        knn.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(knn.best_params_)

        if gridsearch:
            knn = knn.best_estimator_

        # Apply the model
        pred_prob_training_y = knn.predict_proba(train_X)
        pred_prob_test_y = knn.predict_proba(test_X)
        pred_training_y = knn.predict(train_X)
        pred_test_y = knn.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=knn.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=knn.classes_)
        if save_model:
            # save the model to disk
            filename = 'final_' + str(inspect.stack()[0][3]) + '_model_BCI.sav'
            pickle.dump(knn, open(filename, 'wb'))

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a decision tree approach for classification upon the training data (with the specified value for
    # the minimum samples in the leaf, and the export path and files if print_model_details=True)
    # and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def decision_tree(self, train_X, train_y, test_X, min_samples_leaf=50, save_model=False, criterion='gini', print_model_details=False, export_tree_path='./figures/crowdsignals_ch7_classification/', export_tree_name='tree.dot', gridsearch=True):
        # Create the model
        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'criterion':['gini', 'entropy']}]
            dtree = GridSearchCV(DecisionTreeClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            dtree = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, criterion=criterion)

        # Fit the model
        dtree.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(dtree.best_params_)

        if gridsearch:
            dtree = dtree.best_estimator_

        # Apply the model
        pred_prob_training_y = dtree.predict_proba(train_X)
        pred_prob_test_y = dtree.predict_proba(test_X)
        pred_training_y = dtree.predict(train_X)
        pred_test_y = dtree.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=dtree.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=dtree.classes_)

        if print_model_details:
            ordered_indices = [i[0] for i in sorted(enumerate(dtree.feature_importances_), key=lambda x:x[1], reverse=True)]
            print('Feature importance decision tree:')
            for i in range(0, len(dtree.feature_importances_)):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(dtree.feature_importances_[ordered_indices[i]])
            if not (os.path.exists(export_tree_path)):
                os.makedirs(str(export_tree_path))
            tree.export_graphviz(dtree, out_file=str(export_tree_path) + '/' + export_tree_name, feature_names=train_X.columns, class_names=dtree.classes_)
        if save_model:
            # save the model to disk
            filename = 'final_' + str(inspect.stack()[0][3]) + '_model_BCI.sav'
            pickle.dump(dtree, open(filename, 'wb'))

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a naive bayes approach for classification upon the training data
    # and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def naive_bayes(self, train_X, train_y, test_X, save_model=False):
        # Create the model
        nb = GaussianNB()
        
        train_y = train_y.values.ravel()
        # Fit the model
        nb.fit(train_X, train_y)

        # Apply the model
        pred_prob_training_y = nb.predict_proba(train_X)
        pred_prob_test_y = nb.predict_proba(test_X)
        pred_training_y = nb.predict(train_X)
        pred_test_y = nb.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=nb.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=nb.classes_)
        if save_model:
            # save the model to disk
            filename = 'final_' + str(inspect.stack()[0][3]) + '_model_BCI.sav'
            pickle.dump(nb, open(filename, 'wb'))

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y

    # Apply a random forest approach for classification upon the training data (with the specified value for
    # the minimum samples in the leaf, the number of trees, and if we should print some of the details of the
    # model print_model_details=True) and use the created model to predict the outcome for both the
    # test and training set. It returns the categorical predictions for the training and test set as well as the
    # probabilities associated with each class, each class being represented as a column in the data frame.
    def random_forest(self, train_X, train_y, test_X, n_estimators=10, min_samples_leaf=5, criterion='gini', print_model_details=False, gridsearch=True, save_model=False):

        if gridsearch:
            tuned_parameters = [{'min_samples_leaf': [2, 10, 50, 100, 200],
                                 'n_estimators':[10, 50, 100],
                                 'criterion':['gini', 'entropy']}]
            rf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring='accuracy')
        else:
            rf = RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, criterion=criterion)

        # Fit the model

        rf.fit(train_X, train_y.values.ravel())

        if gridsearch and print_model_details:
            print(rf.best_params_)

        if gridsearch:
            rf = rf.best_estimator_

        pred_prob_training_y = rf.predict_proba(train_X)
        pred_prob_test_y = rf.predict_proba(test_X)
        pred_training_y = rf.predict(train_X)
        pred_test_y = rf.predict(test_X)
        frame_prob_training_y = pd.DataFrame(pred_prob_training_y, columns=rf.classes_)
        frame_prob_test_y = pd.DataFrame(pred_prob_test_y, columns=rf.classes_)

        if print_model_details:
            ordered_indices = [i[0] for i in sorted(enumerate(rf.feature_importances_), key=lambda x:x[1], reverse=True)]
            print('Top 20 feature importances random forest:')
            for i in range(0, 20):
                print(train_X.columns[ordered_indices[i]], end='')
                print(' & ', end='')
                print(rf.feature_importances_[ordered_indices[i]])
        
        if save_model:
            # save the model to disk
            filename = 'final_' + str(inspect.stack()[0][3]) + '_model_BCI.sav'
            pickle.dump(rf, open(filename, 'wb'))

        return pred_training_y, pred_test_y, frame_prob_training_y, frame_prob_test_y