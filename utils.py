import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt



def choose_threshold_and_evaluate(model_pred, y_validation):
    precision, recall, thresholds = precision_recall_curve(y_validation, model_pred)
    f_score = 2 * precision * recall / (precision + recall)

    # Choosing the best threshold which maximizes F-score
    best_score = np.argmax(f_score)

    # making 0-1 predictions with the use of optimised threshold value
    class_pred = np.where(model_pred > thresholds[best_score], 1, 0)

    return thresholds[best_score], f_score[best_score], class_pred


def plot_roc(y_test, prob, model_name=""):

    fpr, recall, tresh = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)

    plt.plot(fpr, recall, label="{} AUC = {}".format(model_name, auc.round(2)))
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel("1 - Precision")
    plt.ylabel("Recall")
    plt.title("ROC Curve")
    plt.legend()