import argparse
from loguru import logger
import os
import pickle
import sklearn.metrics
import femr.labelers
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run femr labeling")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to femr database")
    parser.add_argument("--path_to_output_dir", required=True, type=str, help="Path to save labeles")
    args = parser.parse_args()
    
    database = femr.datasets.PatientDatabase(args.path_to_database)

    if False:
        labeled_patients = femr.labelers.load_labeled_patients(os.path.join(args.path_to_output_dir, "labeled_patients.csv"))
        subsampled = femr.labelers.load_labeled_patients(os.path.join(args.path_to_output_dir, "subsample_labeled_patients.csv"))

        positive_weight = np.sum(subsampled.as_numpy_arrays()[1]) /  np.sum(labeled_patients.as_numpy_arrays()[1])
        negative_weight = np.sum(~subsampled.as_numpy_arrays()[1]) /  np.sum(~labeled_patients.as_numpy_arrays()[1])
    else:
        positive_weight = 1
        negative_weight = 0.004052736131203913


    logger.info(f"Positive weight: {positive_weight}, Negative weight: {negative_weight}")

    for model in ['logistic', 'gbm', 'motor']:
        logger.info(f"Evaluating {model}")
        if model != 'motor':
            path = os.path.join(args.path_to_output_dir, f'{model}_predictions.pkl')
        else:
            path = os.path.join(args.path_to_output_dir, f'motor_results/predictions.pkl')

        with open(path, 'rb') as f:
            predictions, patient_ids, labels, label_times = pickle.load(f)

            weights = np.where(labels, 1 / positive_weight, 1 / negative_weight)

            logger.info(f"Modeling prevalence: {np.mean(labels)}")
            logger.info(f"Real prevalence: {np.sum(labels * weights) / np.sum(weights)}")

            val_start = 70
            test_start = 85
            split_seed = 97
            hashed_pids = np.array([database.compute_split(split_seed, pid) for pid in patient_ids])

            train_mask = hashed_pids < val_start
            valid_mask = np.logical_and(hashed_pids >= val_start, hashed_pids < test_start)
            test_mask = hashed_pids >= test_start

            logger.info(f"Num train: {sum(train_mask)} {np.sum(labels[train_mask])}")
            logger.info(f"Num valid: {sum(valid_mask)} {np.sum(labels[valid_mask])}")
            logger.info(f"Num test: {sum(test_mask)} {np.sum(labels[test_mask])}")

            logger.info(f"AUROC: {sklearn.metrics.roc_auc_score(labels[test_mask], predictions[test_mask], sample_weight=weights[test_mask])}")
            logger.info(f"AUPRC: {sklearn.metrics.average_precision_score(labels[test_mask], predictions[test_mask], sample_weight=weights[test_mask])}")

            aurocs = []
            auprcs = []

            for _ in range(1000):
                pred_sample, label_sample, weight_sample = sklearn.utils.resample(predictions[test_mask], labels[test_mask], weights[test_mask])
                aurocs.append(sklearn.metrics.roc_auc_score(label_sample, pred_sample, sample_weight=weight_sample))
                auprcs.append(sklearn.metrics.average_precision_score(label_sample, pred_sample, sample_weight=weight_sample))

            logger.info(f"AUROC 95%: {np.quantile(aurocs, [0.025, 0.975])}")
            logger.info(f"AUPRC 95%: {np.quantile(auprcs, [0.025, 0.975])}")

