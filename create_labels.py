import argparse
import datetime
import os
import pickle
import json
import csv
import numpy as np
from loguru import logger
from typing import Any, Callable, List, Optional, Set, Tuple
import collections

import random
import femr.labelers
import femr.datasets
import femr.labelers.omop

class PADLabeler(femr.labelers.TimeHorizonEventLabeler):
    def __init__(self, ontology, time_horizon):
        self.time_horizon = time_horizon
        self.pad_codes = list(femr.labelers.omop.map_omop_concept_codes_to_femr_codes(ontology, {'SNOMED/840580004'}, is_ontology_expansion=True))

        super().__init__()

    def get_prediction_times(self, patient):
        outpatient_visit_times = set()
        
        first_non_birth = None

        birth = patient.events[0].start

        for event in patient.events:
            if event.start != birth and first_non_birth is None:
                first_non_birth = event.start

            if event.code in self.pad_codes:
                break

            if event.code == "Visit/OP" and first_non_birth is not None and (event.start - first_non_birth).days > 365:
                outpatient_visit_times.add(event.start - datetime.timedelta(days=1))

        return sorted(list(outpatient_visit_times))

    def get_time_horizon(self):
        return self.time_horizon

    def allow_same_time_labels(self):
        return False

    def get_outcome_times(self, patient):
        outcome_times = set()

        for event in patient.events:
            if event.code in self.pad_codes:
                outcome_times.add(event.start)
                
        return sorted(list(outcome_times))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run femr labeling")
    parser.add_argument("--path_to_database", required=True, type=str, help="Path to femr database")
    parser.add_argument("--path_to_output_dir", required=True, type=str, help="Path to save labeles")
    parser.add_argument("--num_threads", type=int, help="The number of threads to use", default=1, )
    parser.add_argument("--use_sample", action=argparse.BooleanOptionalAction, help="Label a sample instead of the whole database", default=False)

    args = parser.parse_args()

    PATH_TO_PATIENT_DATABASE = args.path_to_database
    PATH_TO_OUTPUT_DIR = args.path_to_output_dir
    NUM_THREADS: int = args.num_threads

    # Logging
    path_to_log_file: str = os.path.join(PATH_TO_OUTPUT_DIR, 'labels_info.log')
    if os.path.exists(path_to_log_file):
        os.remove(path_to_log_file)
    logger.add(path_to_log_file, level="INFO")  # connect logger to file
    logger.info(f"Loading patient database from: {PATH_TO_PATIENT_DATABASE}")
    logger.info(f"Saving output to: {PATH_TO_OUTPUT_DIR}")
    logger.info(f"# of threads: {NUM_THREADS}")
    with open(os.path.join(PATH_TO_OUTPUT_DIR, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    # create directories to save files
    PATH_TO_SAVE_LABELED_PATIENTS: str = os.path.join(PATH_TO_OUTPUT_DIR, "labeled_patients.csv")
    PATH_TO_SAVE_SUBSAMPLE_LABELED_PATIENTS: str = os.path.join(PATH_TO_OUTPUT_DIR, "subsample_labeled_patients.csv")
    os.makedirs(PATH_TO_OUTPUT_DIR, exist_ok=True)

    # Load PatientDatabase + Ontology
    logger.info(f"Start | Load PatientDatabase")
    database = femr.datasets.PatientDatabase(PATH_TO_PATIENT_DATABASE)
    ontology = database.get_ontology()
    logger.info(f"Finish | Load PatientDatabase")

    labeler = PADLabeler(ontology, femr.labelers.TimeHorizon(datetime.timedelta(minutes=1), datetime.timedelta(days=60)))

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
    logger.info("LabeledPatient stats:\n"
                f"Total # of patients = {labeled_patients.get_num_patients()}\n"
                f"Total # of labels = {labeled_patients.get_num_labels()}\n"
                f"Total # of positives = {np.sum(labeled_patients.as_numpy_arrays()[1])}")

    subsampled = femr.labelers.subsample_to_prevalence(labeled_patients, 0.2, seed=97)
    
    subsampled.save(PATH_TO_SAVE_SUBSAMPLE_LABELED_PATIENTS)
    logger.info("Finish | Subsampled label patients")
    logger.info("Subsampled LabeledPatient stats:\n"
                f"Total # of patients = {subsampled.get_num_patients()}\n"
                f"Total # of labels = {subsampled.get_num_labels()}\n"
                f"Total # of positives = {np.sum(subsampled.as_numpy_arrays()[1])}")

    logger.info("Done!")