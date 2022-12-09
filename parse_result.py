import json
from pathlib import Path
from collections import defaultdict

import pandas as pd


def parse_hyperparameters(folder_path, result_file):
    results = defaultdict(list)
    for experiment_path in Path(folder_path).iterdir():
        hparams = experiment_path.name
        results['finetune'].append('finetune' in hparams)
        for group in hparams.replace('-finetune', '').split('-'):
            hparam = '_'.join(group.split('_')[:-1])
            value = group.split('_')[-1]
            results[hparam].append(value)

        for key, val in parse_experiment_folder(experiment_path, result_file).items():
            results[key].append(val)

    return pd.DataFrame(results)


def parse_experiment_folder(folder_path, result_file):
    results = defaultdict(list)
    for result_path in Path(folder_path).rglob(f'{result_file}'):
        dataset = result_path.parents[0].name
        with open(result_path, 'r') as fin:
            result = json.load(fin)['best_val_test'] * 100
        results[dataset].append(result)

    datasets = ['mnist_m', 'svhn', 'syn', 'mnist']
    results = {dataset: sum(results[dataset]) / len(results[dataset]) for dataset in datasets}
    results['avg'] = sum(results.values()) / len(results)
    return results


baseline = parse_experiment_folder('./result/baseline', 'phase1.json')
grad_reverse = parse_experiment_folder('./result/grad_reverse', 'phase1.json')
grad_reverse2 = parse_experiment_folder('./result/grad_reverse-5e-2', 'phase1.json')
method1 = parse_hyperparameters('./result/method1', 'phase1.json')
method2 = parse_hyperparameters('./result/method3', 'phase1.json')
method3 = parse_hyperparameters('./result/method3', 'phase2.json')
print(baseline)
print(grad_reverse)
print(grad_reverse2)
method1.to_csv('result/method1.csv', index=None)
method2.to_csv('result/method2.csv', index=None)
method3.to_csv('result/method3.csv', index=None)
