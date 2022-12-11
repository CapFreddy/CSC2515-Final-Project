import pickle
import argparse
import numpy as np
import pandas as pd
import pandas as pd
import seaborn as sns
from sklearn import svm
from typing import Union, List
import matplotlib.pyplot as plt
from collections import OrderedDict
from torch.utils.data import DataLoader
from pathlib2 import Path
from tqdm import tqdm

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, mean_squared_error, classification_report, confusion_matrix, precision_recall_curve, PrecisionRecallDisplay, RocCurveDisplay
from joblib import dump, load

datasets = ['mnist', 'mnist_m', 'svhn', 'syn']

from src.DGDataset import DGDataset

def get_performance_metrics(predictions: np.ndarray, labels: np.ndarray):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    mse = mean_squared_error(labels, predictions)
    cm = confusion_matrix(labels, predictions)
    classification_rpt = classification_report(
        labels, predictions, output_dict=True)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "mse": mse,
        "cm": cm,
        "classification_rpt": classification_rpt,
        "classification_rpt_df": pd.DataFrame(classification_rpt).transpose()
    }


def load_dataset(datasets: List[str], mode: str = 'train'):
    dataset = DGDataset(datasets, mode=mode)
    dataloader = DataLoader(dataset, batch_size=100)
    data, labels, domains = [], [], []
    for d, label, domain in dataloader:
        data.extend(d.numpy())
        labels.extend(label.numpy())
        domains.extend(domain.numpy())
    data = np.array(data)
    if len(data.shape) == 4:
        # has a color channel dimension
        # flatten each image to a vector
        data = data.reshape(len(data), np.prod(data.shape[1:]))
    return data, labels, domains


def get_svm_performance(train_data, train_labels, test_data, test_labels) -> dict:
    svm_model = SVC(C=10, decision_function_shape='ovo',
                    gamma='scale', kernel='rbf').fit(train_data, train_labels)
    # print(f"Accuracy: {round(accuracy_score(svm_model.predict(test_data), test_labels) * 100, 2)}%")
    predictions = svm_model.predict(test_data)
    svm_performance = get_performance_metrics(predictions, test_labels)
    return svm_performance


def get_decision_tree_performance(train_data, train_labels, test_data, test_labels) -> dict:
    dt_clf = DecisionTreeClassifier(random_state=args.seed)
    dt_clf.fit(train_data, train_labels)
    predictions = dt_clf.predict(test_data)
    dt_performance = get_performance_metrics(predictions, test_labels)
    return dt_performance


def get_random_forest_performance(train_data, train_labels, test_data, test_labels) -> dict:
    rf_clf = RandomForestClassifier(random_state=args.seed, n_estimators=100, criterion="entropy", max_features='sqrt')
    rf_clf.fit(train_data, train_labels)
    predictions = rf_clf.predict(test_data)
    rf_performance = get_performance_metrics(predictions, test_labels)
    return rf_performance


def get_adaboost_performance(train_data, train_labels, test_data, test_labels) -> dict:
    adaboost_dt_base_estimator = DecisionTreeClassifier(max_depth=10)
    adaboost_clf = AdaBoostClassifier(
        n_estimators=100, random_state=args.seed, base_estimator=adaboost_dt_base_estimator)
    adaboost_clf.fit(train_data, train_labels)
    predictions = adaboost_clf.predict(test_data)
    adaboost_performance = get_performance_metrics(predictions, test_labels)
    return adaboost_performance


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_domain", choices=datasets,
                        required=True, help="Choose Target Domain")
    parser.add_argument("-o", "--output", help="output csv file path")
    parser.add_argument("-s", "--seed", type=int, help="random seed")
    args = parser.parse_args()
    stats = []
    np.random.seed(args.seed)
    # set up dataset
    target_domain = args.target_domain
    train_datasets = datasets.copy()
    train_datasets.remove(target_domain)
    train_data, train_labels, train_domains = load_dataset(
        train_datasets, mode='train')
    val_data, val_labels, val_domains = load_dataset(
        train_datasets, mode='val')
    test_data, test_labels, test_domains = load_dataset(
        [target_domain], mode='test')

    pbar = tqdm(total=4)
    pbar.set_postfix({"method": "svm"})

    svm_performance = get_svm_performance(
        train_data, train_labels, test_data, test_labels)
    stats.append({
        "method": "svm",
        "seed": args.seed,
        "target_domain": target_domain,
        "accuracy": svm_performance['accuracy'],
        "precision": svm_performance['precision'],
        "recall": svm_performance['recall'],
        "mse": svm_performance['mse']
    })
    pbar.update()

    # pbar.set_postfix({"method": "Decision Tree"})
    # dt_performance = get_decision_tree_performance(
    #     train_data, train_labels, test_data, test_labels)
    # stats.append({
    #     "method": "decision_tree",
    #     "seed": args.seed,
    #     "target_domain": target_domain,
    #     "accuracy": dt_performance['accuracy'],
    #     "precision": dt_performance['precision'],
    #     "recall": dt_performance['recall'],
    #     "mse": dt_performance['mse']
    # })
    # pbar.update()

    pbar.set_postfix({"method": "Random Forest"})
    rf_performance = get_random_forest_performance(
        train_data, train_labels, test_data, test_labels)
    stats.append({
        "method": "random_forest",
        "seed": args.seed,
        "target_domain": target_domain,
        "accuracy": rf_performance['accuracy'],
        "precision": rf_performance['precision'],
        "recall": rf_performance['recall'],
        "mse": rf_performance['mse']
    })
    pbar.update()

    pbar.set_postfix({"method": "AdaBoost"})
    adaboost_performance = get_adaboost_performance(
        train_data, train_labels, test_data, test_labels)
    stats.append({
        "method": "adaboost",
        "seed": args.seed,
        "target_domain": target_domain,
        "accuracy": adaboost_performance['accuracy'],
        "precision": adaboost_performance['precision'],
        "recall": adaboost_performance['recall'],
        "mse": adaboost_performance['mse']
    })
    pbar.update()
    df = pd.DataFrame(stats)
    print(df)
    df.to_csv(args.output)
