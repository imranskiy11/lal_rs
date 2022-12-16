from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, recall_score, f1_score
from termcolor import colored


def bin_metric_score_output(true_labels, pred_labels):
    # print(colored(f'accuracy_score  : {round(accuracy_score(true_labels, pred_labels), 3)}', 'blue'))
    print(colored(f'precision_score : {round(precision_score(true_labels, pred_labels), 3)}', 'blue'))
    print(colored(f'recall_score    : {round(recall_score(true_labels, pred_labels), 3)}', 'blue'))
    print(colored(f'f1_score        : {round(f1_score(true_labels, pred_labels), 3)}', 'blue'))
    print(colored(f'roc_auc_score   : {round(roc_auc_score(true_labels, pred_labels), 3)}', 'blue'))