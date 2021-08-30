from sklearn import metrics
import pandas as pd
import numpy as np
import math

# Class for evaluation metrics of classification problems.
class ClassificationEvaluation:

    # Returns the accuracy given the true and predicted values.
    def accuracy(self, y_true, y_pred):
        return metrics.accuracy_score(y_true, y_pred)

    # Returns the precision given the true and predicted values.
    # Note that it returns the precision per class.
    def precision(self, y_true, y_pred):
        return metrics.precision_score(y_true, y_pred, average=None)

    # Returns the recall given the true and predicted values.
    # Note that it returns the recall per class.
    def recall(self, y_true, y_pred):
        return metrics.recall_score(y_true, y_pred, average=None)

    # Returns the f1 given the true and predicted values.
    # Note that it returns the recall per class.
    def f1(self, y_true, y_pred):
        return metrics.f1_score(y_true, y_pred, average='macro')

    # Returns the area under the curve given the true and predicted values.
    # Note: we expect a binary classification problem here(!)
    def auc(self, y_true, y_pred_prob):
        return metrics.roc_auc_score(y_true, y_pred_prob)

    # Returns the confusion matrix given the true and predicted values.
    def confusion_matrix(self, y_true, y_pred, labels):
        return metrics.confusion_matrix(y_true, y_pred, labels=labels)