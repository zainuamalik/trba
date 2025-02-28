# ============================
# EXECUTE TRAINING WITH OPTUNA
# ============================

import subprocess
import optuna
import joblib 
import os
from pathlib import Path
import argparse
import re

def extract_last_value(pattern, log_content):
    """
    Extracts the last value from the log content using the provided regular expression pattern.
    Returns None if no matches are found.
    """
    matches = re.findall(pattern, log_content)  # Find all matches for the pattern
    if matches:
        try:
            return float(matches[-1])  # Convert the last match to a float
        except ValueError:
            print(f'Failed to convert extracted value to float: {matches[-1]}')
    return None

def is_duplicate_hyperparams(study, current_params, past_studies=None):
    """
    Checks if the current hyperparameters match any of the successful trials
    from the current or past studies.
    """
    # Check against the current study
    for trial in study.get_trials(states=[optuna.trial.TrialState.COMPLETE]):
        if trial.value is not None and trial.value > 0:  # Successful trial
            if trial.params == current_params:
                return True

    # Check against past studies if provided
    if past_studies:
        for past_study in past_studies:
            for trial in past_study.get_trials(states=[optuna.trial.TrialState.COMPLETE]):
                if trial.value is not None and trial.value > 0:  # Successful trial
                    if trial.params == current_params:
                        return True

    return False

def suggest_hyperparameters(trial):
    """
    Suggest hyperparameters for the trial.
    """
    # Define hyperparameters
    hyperparams = {
        'batch_size': trial.suggest_categorical("batch_size", [20, 40, 80]),
        'lr': trial.suggest_float("lr", 0.1, 1, log=True),
        'num_iter': trial.suggest_int("num_iter", 200, 2000, step=200),
        # 'scheduler': trial.suggest_categorical("scheduler", ["step", "cosine", "plateau"]),
        # 'num_control_points': trial.suggest_int("num_fiducial", 6, 16, step=2),
        # 'hidden_units': trial.suggest_int("hidden_size", 256, 1024, step=256),
        # 'imgH': trial.suggest_categorical("imgH", [32, 48, 64]),
        # 'imgW': trial.suggest_categorical("imgW", [100, 128, 160]),
        # 'weight_decay': trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        # 'grad_clip': trial.suggest_float("grad_clip", 0.1, 5.0),
        # 'optimizer': trial.suggest_categorical("optimizer", ["Adam", "SGD", "AdamW"])
        
        # For STEP scheduler
        # "step_size": trial.suggest_int("step_size", 5, 15, step=5),
        # "gamma": trial.suggest_float("gamma", 0.1, 0.9),
        
        # For COSINE scheduler
        # "T_max": trial.suggest_int("T_max", 50, 200, step=50),
        
        # For REDUCELRONPLATEAU scheduler
        # "mode": trial.suggest_categorical("mode", ["min", "max"]),
        # "factor": trial.suggest_float("factor", 0.1, 0.5),
        # "patience": trial.suggest_int("patience", 5, 20),
        # "verbose": trial.suggest_categorical("verbose", [True, False]),
    }
    
    return hyperparams

def run_training_script(trial, hyperparams):
    """
    Executes the training script with parameters from the Optuna trial.
    """
    # Construct the command to run the training script
    cmd = [
        "python",  # Use Python to run the script
        "train_capt.py",  # Script name
        "--exp_name", f"trial_{trial.number}",
        "--train_data", "lmdb_aug_trainH",   
        "--valid_data", "lmdb_combined_testH",   
        "--Transformation", "TPS",
        "--FeatureExtraction", "ResNet",
        "--SequenceModeling", "BiLSTM",
        "--Prediction", "Attn",
        "--valInterval", "5",
        "--workers", "0",
        "--select_data", "/",
        "--batch_ratio", "1.0",
        "--data_filtering_off",
        "--saved_model", "optuna/Pre-trained-TRBA/TPS-ResNet-BiLSTM-Attn.pth"
    ] 
    
    """ # Add scheduler arguments to the command
    if hyperparams["scheduler"] == "step":
        cmd += ["--scheduler", "step", "--step_size", str(hyperparams["step_size"]), "--gamma", str(hyperparams["gamma"])]
    
    elif hyperparams["scheduler"] == "cosine":
        cmd += ["--scheduler", "cosine", "--T_max", str(hyperparams["T_max"])]
    
    elif hyperparams["scheduler"] == "plateau":
        cmd += [
            "--scheduler", "plateau",
            "--mode", hyperparams["mode"],
            "--factor", str(hyperparams["factor"]),
            "--patience", str(hyperparams["patience"]),
            "--verbose", str(hyperparams["verbose"]),
        ] """
    
    # Dynamically add hyperparameters
    for key, value in hyperparams.items():
        if isinstance(value, bool):  # Handle boolean flags
            if value:
                cmd.append(f"--{key}")
        else:
            cmd += [f"--{key}", str(value)]
    
    metrics = {
        "accuracy": 0.0,
        "duration": None,
        }
    
    try:
        # Run the training script
        subprocess.run(cmd, check=True)

        # Extract validation accuracy from the log file
        log_path = f"saved_models/trial_{trial.number}/log_train.txt"
    
        
        try:
            with open(log_path, "r") as f:
                log_content = f.read()
                
                metrics["accuracy"] = extract_last_value(r"Best_accuracy\s*:\s*([\d.]+)", log_content)
                metrics["duration"] = extract_last_value(r"Elapsed_time\s+:\s+([\d.]+)\s", log_content)
                if metrics["duration"] is not None:
                    trial.set_user_attr("time_taken", metrics["duration"])  # Save duration in trial attributes
                    
        except FileNotFoundError as e:
            print(f'Log file not found: {log_path}. Error: {e}')
        
        except Exception as e:
            print(f"Error parsing log file for trial {trial.number}: {e}")
        
        return {"hyperparams": hyperparams, "status": "success", "accuracy": metrics['accuracy'], "time_taken": metrics['duration']}
        
    except Exception as e:
        # Handle errors during training
        error_log_path = f"saved_models/trial_{trial.number}/error_log.txt"
        
        with open(error_log_path, "w") as error_file:
            
            error_file.write(f"Error during trial {trial.number}:\n\t {str(e)}\n\n\t")
            error_file.write("Hyperparameters used:\n")
            
            for key, value in hyperparams.items():
                error_file.write(f"{key}: {value}\n")
                
        print(f"Error during trial {trial.number}: {e}")
        
        return {"status": "failed", "error": str(e), "time_taken": metrics["duration"]}


# ============================
# Define Objective Function
# ============================

def objective(trial):
    """
    Objective function for Optuna to optimize hyperparameters.
    """
    
    while True:
        
        # Suggest hyperparameters
        hyperparams = suggest_hyperparameters(trial)

        # Check for duplicates
        if not is_duplicate_hyperparams(trial.study, hyperparams, past_studies=past_studies):
            
            # Directory for the current trial
            trial_dir = f"saved_models/trial_{trial.number}"
            os.makedirs(trial_dir, exist_ok=True)

            # Save hyperparameters to a text file
            with open(os.path.join(trial_dir, "hyperparameters.txt"), "w") as f:
                for key, value in hyperparams.items():
                    f.write(f"{key}: {value}\n")
            break  # Exit loop if hyperparameters are not duplicates
        
        else:
            print(f"Duplicate hyperparameters detected, re-suggesting for trial {trial.number}...")
            with open("duplicates.txt", "a") as log_file:
                log_file.write(f"Trial {trial.number} duplicates: {hyperparams}\n")
            continue        

    # Proceed with training and evaluation
    trial_result = run_training_script(trial, hyperparams)

    if trial_result["status"] == "success":
        trial.set_user_attr("time_taken", trial_result["time_taken"])  # Save duration for summary
        return trial_result["accuracy"]
    else:
        return 0.0

# ============================
# Main Script
# ============================

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_trials", type=int, default=5, help="Number of Optuna trials.")
    parser.add_argument("--direction", type=str, default="maximize", choices=["maximize", "minimize"], help="Optimization direction.")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel processes.")
    parser.add_argument("--past_study_files", nargs="*", help="Paths to previous study files (optional).")

    args = parser.parse_args()
    
    # Load past studies if provided
    past_studies = []
    if args.past_study_files:
        for study_file in args.past_study_files:
            try:
                past_studies.append(joblib.load(study_file))
                print(f"Loaded past study from '{study_file}'.")
            except Exception as e:
                print(f"Failed to load study from '{study_file}': {e}")
    
    # Create an Optuna study (maximize validation accuracy)
    study = optuna.create_study(direction=args.direction)

    # Run the optimization
    print("Starting hyperparameter tuning...")
    study.optimize(objective, n_trials=args.n_trials, n_jobs=args.n_jobs) 

    trials_summary = []
    
    for trial in study.trials:
        
        trial_info = {
            "trial_number": trial.number,
            "status": "success" if trial.value else "failed",
            "hyperparameters": trial.params if trial.value else None,
            "accuracy": trial.value if trial.value else 0.0,
            "error": trial.user_attrs.get("error", ""),
            "time_taken": trial.user_attrs.get("time_taken", None),
        }
        trials_summary.append(trial_info)
    
    # Print the best trial
    print("\nBest trial:")
    print(f"  Value (Accuracy): {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")

    # Save the study results
    print("\nSaving study results...")
    results_path = "saved_models/Cycle"
    
    os.makedirs(results_path, exist_ok=True)

    study_file = os.path.join(results_path, "study.pkl")
    csv_file = os.path.join(results_path, "tuning_results.csv")

    if os.path.exists(study_file):
        print(f"Warning: {study_file} already exists and will be overwritten.")
    if os.path.exists(csv_file):
        print(f"Warning: {csv_file} already exists and will be overwritten.")

    joblib.dump(study, study_file)
    print(f"Study object saved to '{results_path}/study.pkl'")
    
    # Produce summary file for study (Info about each trial)
    summary_file = os.path.join(results_path, "summary.txt")
    
    with open(summary_file, "w") as f:
        
        f.write("Model Tuning Summary\n")
        f.write(f"Total Trials: \n  {len(trials_summary)}\n")
        f.write(f"Successful Trials: \n  {len([t for t in trials_summary if t['status'] == 'success'])}\n")
        f.write(f"Failed Trials: \n  {len([t for t in trials_summary if t['status'] == 'failed'])}\n")
        f.write(f"Best Trial: \n  Trial {study.best_trial.number}, Best Accuracy Achieved: {study.best_trial.value}\n\n")
        
        for t in trials_summary:
            
            f.write(f"Trial {t['trial_number']}:\n")
            f.write(f"  Status: {t['status']}\n")
            if t["status"] == "success":
                f.write(f"  Best Accuracy: {t['accuracy']}\n")
                f.write(f"  Hyperparameters: {t['hyperparameters']}\n")
            else:
                f.write(f"  Error: {t['error']}\n")
                
            f.write(f"  Time Taken: {t['time_taken'] if t['time_taken'] is not None else 'N/A'} seconds\n")
            f.write("\n")

    study.trials_dataframe().to_csv(csv_file, index=False)
    print(f"Results saved to '{results_path}/tuning_results.csv'")