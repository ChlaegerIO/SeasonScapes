import os
from tqdm import tqdm
from pathlib import Path
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def get_last_value(log_dir, tag):
    # Create an EventAccumulator to read the event files
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()

    # Check if the tag exists
    if tag not in event_acc.Tags()['scalars']:
        print(f"Tag '{tag}' not found in the event files for log directory '{log_dir}'.")
        return None

    # Extract the data for the specified tag
    events = event_acc.Scalars(tag)
    if not events:
        print(f"No data found for tag '{tag}' in log directory '{log_dir}'.")
        return None

    # Get the last value
    last_value = events[-1].value
    return last_value

def extract_last_values_for_multiple_runs(log_dirs, tag):
    last_values = {}
    for log_dir in tqdm(log_dirs):
        last_value = get_last_value(log_dir, tag)
        if last_value is not None:
            last_values[log_dir] = last_value
    return last_values

# Specify the log directories and the tag you want to extract
tensorboard_logs_dir = 'data_engine/Logs/poseOptim_v2'
tag = 'train/matching_loss'

log_dirs = [os.path.join(tensorboard_logs_dir, folder) for folder in os.listdir(tensorboard_logs_dir) if os.path.isdir(os.path.join(tensorboard_logs_dir, folder))]

# Get the last values for each run
print(f"Get last values for tag {tag} in {tensorboard_logs_dir}")
last_values = extract_last_values_for_multiple_runs(log_dirs, tag)

# Print the last values for each run
txt_file = []
txt_file.append(f"Directory: Last {tag} value")
for log_dir, last_value in last_values.items():
    last_value = round(last_value, 1) 
    cam_id = Path(log_dir).stem.split('_')[1]
    txt_file.append(f"{cam_id} & {last_value}")
    print(f"{cam_id} & {last_value}")

mean_value = np.array(list(last_values.values())).mean()
std_value = np.array(list(last_values.values())).std()

txt_file.append("-------------------")
txt_file.append(f"Mean: {mean_value:.2f} +/- {std_value:.2f}")

with open(os.path.join(tensorboard_logs_dir, 'data_export_last.txt'), 'w') as f:
    for line in txt_file:
        f.write(line + '\n')

print(f"Last values exported to {tensorboard_logs_dir}/data_export_last.txt")
