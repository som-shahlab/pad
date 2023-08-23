import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run femr labeling")
    parser.add_argument(
        "--path_to_database", required=True, type=str, help="Path to femr database"
    )
    parser.add_argument(
        "--path_to_output_dir", required=True, type=str, help="Path to save labeles"
    )
    parser.add_argument(
        "--path_to_motor", required=True, type=str, help="Path to the motor model"
    )

    args = parser.parse_args()

    task_batches = os.path.join(args.path_to_output_dir, "tte_motor_batches")
    labels = os.path.join(args.path_to_output_dir, "subsample_labeled_patients.csv")

    command = f"clmbr_create_batches {task_batches} --data_path {args.path_to_database} --task labeled_patients --labeled_patients_path {labels} --val_start 70 --dictionary_path {args.path_to_motor}/dictionary --is_hierarchical --batch_size 262144"
    os.system(command)

    motor_results = os.path.join(args.path_to_output_dir, "tte_motor_results")

    command = f"clmbr_train_linear_probe {motor_results} --data_path {args.path_to_database} --model_dir {args.path_to_motor}/model --batches_path {task_batches}"
    os.system(command)
