import os
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
import hyperparameter
import data_loader

def assign_value(detector_info, shape):
    """
    Assign detector values to additional elements tensor based on channel information.
    """
    # Define the required keys and their mapping
    required_keys = [
        "ant_orientation_phi",
        "ant_orientation_theta",
        "ant_position_x",
        "ant_position_y",
        "ant_position_z",
        "ant_rotation_phi",
        "ant_rotation_theta",
        "ant_type",
    ]
    ant_type_mapping = {
        "createLPDA_100MHz_InfFirn": 0,
        "RNOG_vpol_4inch_center_n1.73": 1,
        # Add more mappings as needed
    }

    # Initialize additional elements tensor with zeros
    additional_elements = torch.zeros(shape[0], shape[1], len(required_keys))

    # Populate tensor with detector information
    channels = detector_info.get("channels", {})
    for channel_id, channel_info in channels.items():
        for idx, key in enumerate(required_keys):
            value = channel_info.get(key)
            if key == "ant_type":
                value = ant_type_mapping.get(value)  
            for i in range(shape[0]):
                additional_elements[i, int(channel_id) - 1, idx] = value
    return additional_elements


# Data configuration
data_type = 'testing'  # Options: 'training', 'validating', 'testing'
label_type = 'nc'      # Options: 'cc', 'nc_cc', 'nc'
batch_size = hyperparameter.batchSize
detector_file_location = '/mnt/hdd/nheyer/dl_playground/sim_file/detector.json'
new_data_location = f"/mnt/hdd/moniba/dl_playground/data/gen2_shallow_baseline/{label_type}/{data_type}"

# Create output directory if it does not exist
os.makedirs(new_data_location, exist_ok=True)

# Define data source based on data type
if data_type == 'training':
    data_location = hyperparameter.data_location_training
    list_of_file_ids = np.arange(hyperparameter.n_files_train, dtype=int)
elif data_type == 'validating':
    data_location = hyperparameter.data_location_validating
    list_of_file_ids = np.arange(hyperparameter.n_files_val, dtype=int)
elif data_type == 'testing':
    data_location = hyperparameter.data_location_testing
    list_of_file_ids = np.arange(hyperparameter.n_files_test, dtype=int)

# Load detector information
with open(detector_file_location, "r") as file:
    detector_info = json.load(file)

# Prepare data loader
prepared_loader = data_loader.Prepare_Dataset(
    file_ids=list_of_file_ids,
    points=hyperparameter.n_events_per_file,
    data_type=f"_{data_type}",
    worker_num=hyperparameter.worker_num
)
loader = DataLoader(prepared_loader, batch_size=batch_size, shuffle=False, num_workers=hyperparameter.worker_num)

# Process and save data
accumulated_data = None
i_file = 0

for batch_idx, batch in enumerate(loader):
    data, direction, energy, flavor = batch

    # Ensure `data` has the expected shape
    assert data.shape[-1] == 512, f"Unexpected last dimension: {data.shape[-1]}"
    
    reshaped_data = data.squeeze(1)  # Reshape to [128, 5, 512]
    additional_elements = assign_value(detector_info, reshaped_data.shape)
    reshaped_data = torch.cat((reshaped_data, additional_elements), dim=-1)

    # Verify new shape
    assert reshaped_data.shape[-1] == 520, f"Unexpected new shape: {reshaped_data.shape}"

    # Accumulate data
    if accumulated_data is None:
        accumulated_data = reshaped_data.numpy()
    else:
        accumulated_data = np.concatenate((accumulated_data, reshaped_data.numpy()), axis=0)

    # Save data if accumulated data reaches the limit
    if accumulated_data.shape[0] >= hyperparameter.n_events_per_file:
        new_data_path = os.path.join(
            new_data_location,
            f"{hyperparameter.data_name}{hyperparameter.interaction}_{data_type}_10k_{i_file}_.npy"
        )
        np.save(new_data_path, accumulated_data[:hyperparameter.n_events_per_file])
        print(f"File {i_file} processed and saved as {new_data_path}.")

        # Prepare for next batch
        i_file += 1
        accumulated_data = accumulated_data[hyperparameter.n_events_per_file:]

# Save remaining data
if accumulated_data.shape[0] > 0:
    new_data_path = os.path.join(
        new_data_location,
        f"{hyperparameter.data_name}{hyperparameter.interaction}_{data_type}_10k_{i_file}_.npy"
    )
    np.save(new_data_path, accumulated_data)
    print(f"File {i_file} processed and saved as {new_data_path}.")
