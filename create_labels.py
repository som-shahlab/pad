import argparse
import collections
import csv
import datetime
import json
import os
import pickle
import random
from typing import Any, Callable, List, Optional, Set, Tuple

import femr.datasets
import femr.labelers
import femr.labelers.omop
import numpy as np
from loguru import logger


class PADSurvivalLabeler(femr.labelers.Labeler):
    def __init__(self, ontology):
        self.required_days = 365
        
    def label(self, patient):
        birth_date = datetime.datetime.combine(patient.events[0].start.date(), datetime.time.min)
        censor_time = patient.events[-1].start
        
        possible_times = []
        first_history = None
        first_code = None
        has_ibd_code = False
        
        for event in patient.events:

            if first_history is None and (event.start - birth_date) > datetime.timedelta(days=10):
                first_history = event.start
            
            if event.omop_table == 'condition_occurrence' and event.source_code == 'K83.01':
                first_code = event.start

            if event.omop_table == 'condition_occurrence' and (
                event.source_code.startswith('K50') or event.source_code.startswith('K51')
            ):
                has_ibd_code = True
            
            if not has_ibd_code:
                continue

            if not event.code.startswith('Visit/'):
                continue

            possib_time = event.end

            if possib_time == censor_time:
                continue

            if True:
                if event.start.year < 2010:
                    continue
            
                if first_history is None or (possib_time - first_history) <= datetime.timedelta(days=self.required_days):
                    continue

            possible_times.append(possib_time)
        
        possible_times = [a for a in possible_times if first_code is None or a < first_code]
        if len(possible_times) == 0:
            return []
        
        selected_time = random.choice(possible_times)
        is_censored = first_code is None
        
        if is_censored:
            event_time = censor_time
        else:
            event_time = first_code

        survival_value = femr.labelers.SurvivalValue(time_to_event=event_time - selected_time, is_censored=is_censored)
        result = [femr.labelers.Label(time=selected_time, value=survival_value)]
        return result
    
    def get_labeler_type(self):
        return "survival"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run femr labeling")
    parser.add_argument(
        "--path_to_database", required=True, type=str, help="Path to femr database"
    )
    parser.add_argument(
        "--path_to_output_dir", required=True, type=str, help="Path to save labeles"
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        help="The number of threads to use",
        default=1,
    )
    parser.add_argument(
        "--use_sample",
        action=argparse.BooleanOptionalAction,
        help="Label a sample instead of the whole database",
        default=False,
    )

    args = parser.parse_args()

    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_OUTPUT_DIR = args.path_to_output_dir
    NUM_THREADS: int = args.num_threads

    # Logging
    path_to_log_file: str = os.path.join(PATH_TO_OUTPUT_DIR, "labels_info.log")
    if os.path.exists(path_to_log_file):
        os.remove(path_to_log_file)
    logger.add(path_to_log_file, level="INFO")  # connect logger to file
    logger.info(f"Loading patient database from: {PATH_TO_PATIENT_DATABASE}")
    logger.info(f"Saving output to: {PATH_TO_OUTPUT_DIR}")
    logger.info(f"# of threads: {NUM_THREADS}")
    with open(os.path.join(PATH_TO_OUTPUT_DIR, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # create directories to save files
    PATH_TO_SAVE_LABELED_PATIENTS: str = os.path.join(
        PATH_TO_OUTPUT_DIR, "labeled_patients.csv"
    )
    PATH_TO_SAVE_SUBSAMPLE_LABELED_PATIENTS: str = os.path.join(
        PATH_TO_OUTPUT_DIR, "subsample_labeled_patients.csv"
    )
    PATH_TO_SAVE_BINARY_LABELED_PATIENTS: str = os.path.join(
        PATH_TO_OUTPUT_DIR, "binary_subsample_labeled_patients.csv"
    )
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    # Load PatientDatabase + Ontology
    logger.info(f"Start | Load PatientDatabase")
    database = femr.datasets.PatientDatabase(PATH_TO_PATIENT_DATABASE)
    ontology = database.get_ontology()
    logger.info(f"Finish | Load PatientDatabase")

    labeler = PADSurvivalLabeler(
        ontology,
    )

    logger.info(f"Start | Label")

    if args.use_sample:
        num_patients = 10_000
    else:
        num_patients = None

    labeled_patients = labeler.apply(
        path_to_patient_database=PATH_TO_PATIENT_DATABASE,
        num_threads=NUM_THREADS,
        num_patients=num_patients,
    )

    labeled_patients.save(PATH_TO_SAVE_LABELED_PATIENTS)
    logger.info("Finish | Label patients")
    logger.info(
        "LabeledPatient stats:\n"
        f"Total # of patients = {labeled_patients.get_num_patients()}\n"
        f"Total # of labels = {labeled_patients.get_num_labels()}\n"
        f"Total # of uncensored = {np.sum(1 - labeled_patients.as_numpy_arrays()[1][:, 1])}"
    )

    total_uncensored = np.sum(labeled_patients.as_numpy_arrays()[1][:, 1] == 0)
    total_censored = np.sum(labeled_patients.as_numpy_arrays()[1][:, 1] == 1)
    total_censored_to_sample = 4 * total_uncensored

    label_pids, label_values, label_times = labeled_patients.as_numpy_arrays()

    print(label_values.dtype)

    random_numbers = np.random.random(size=(len(label_pids),))

    mask = np.logical_or(random_numbers < (total_censored_to_sample / total_censored), label_values[:, 1] == 0)

    sampled_label_pids = label_pids[mask]
    sampled_label_values = label_values[mask, :]
    sampled_label_times = label_times[mask]

    subsampled = femr.labelers.LabeledPatients.load_from_numpy(sampled_label_pids, sampled_label_values, sampled_label_times, "survival")

    subsampled.save(PATH_TO_SAVE_SUBSAMPLE_LABELED_PATIENTS)
    logger.info("Finish | Subsampled label patients")
    logger.info(
        "Subsampled LabeledPatient stats:\n"
        f"Total # of patients = {subsampled.get_num_patients()}\n"
        f"Total # of labels = {subsampled.get_num_labels()}\n"
        f"Total # of uncensored = {np.sum(1 - subsampled.as_numpy_arrays()[1][:, 1])}"
    )

    logger.info("Convert to binary")

    within_time_range = sampled_label_values[:, 0] <= 365 * 24 * 60
    is_censor = sampled_label_values[:, 1] == 1

    mask = ~np.logical_and(is_censor, within_time_range)

    is_true = np.logical_and(~is_censor, within_time_range)
    
    binary_label_pids = sampled_label_pids[mask]
    binary_label_values = is_true[mask]
    binary_label_times = sampled_label_times[mask]

    binary = femr.labelers.LabeledPatients.load_from_numpy(binary_label_pids, binary_label_values, binary_label_times, "boolean")

    binary.save(PATH_TO_SAVE_BINARY_LABELED_PATIENTS)
    logger.info("Finish | Subsampled label patients")
    logger.info(
        "Subsampled LabeledPatient stats:\n"
        f"Total # of patients = {binary.get_num_patients()}\n"
        f"Total # of labels = {binary.get_num_labels()}\n"
        f"Total # of positives = {np.sum(binary.as_numpy_arrays()[1])}"
    )

    logger.info("Done!")
