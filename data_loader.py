import os
import numpy as np
import hyperparameter
import torch
from queue import Queue
from torch.utils.data import Dataset
from typing import Any, Iterable, List, Optional, Tuple, Union
import json
import threading

"""
Prepare_Dataset makes it possible to read large amounts of data without losing too much time during training. 
"""


class Prepare_Dataset(Dataset):
    def __init__(self, file_ids, points=10000, data_type = '_training', worker_num=0, transform=None, target_transform=None, add_xyz=False):

        self.file_ids = file_ids
        self.data = None
        self.direction = None
        self.energy = None
        self.points = points
        self.transform = transform
        self.target_transform = target_transform
        self.data_type = data_type
        self.worker_num = worker_num
        self.add_xyz = add_xyz
        
    def __len__(self):
        return len(self.file_ids)*self.points

    def __getitem__(self, idx):


        worker_info = torch.utils.data.get_worker_info()

        file_idx = idx // self.points

        if np.mod((idx+self.points),self.points) == 0:

            self.data, self.direction, self.energy, self.flavor = self.get_data(file_idx)

        return self.data[idx-self.points*file_idx].to(torch.float32), self.direction[:, idx-self.points*file_idx].to(torch.float32), self.energy[idx-self.points*file_idx].to(torch.float32), self.flavor[idx-self.points*file_idx].to(torch.float32)

    def assign_value(self, shape):

        """
        Assign detector values to additional elements tensor based on channel information.
        """
        # Load detector information
        with open(hyperparameter.detector_file_location, "r") as file:
             detector_info = json.load(file)
        # Define the required keys and their mapping
        required_keys_and_norms = [
            ("ant_orientation_phi", 360),
            ("ant_orientation_theta", 360),
            ("ant_position_x", 10),
            ("ant_position_y", 10),
            ("ant_position_z", 10),
            ("ant_rotation_phi", 360),
            ("ant_rotation_theta", 360),
            ("ant_type", 1),
        ]
        ant_type_mapping = {
            "createLPDA_100MHz_InfFirn": 0,
            "RNOG_vpol_4inch_center_n1.73": 1,
            # Add more mappings as needed
        }

        # Initialize additional elements tensor with zeros
        additional_elements = torch.zeros(shape[0], shape[1], len(required_keys_and_norms))

        # Populate tensor with detector information
        channels = detector_info.get("channels", {})
        for channel_id, channel_info in channels.items():
            for idx, temp in enumerate(required_keys_and_norms):
                key, norm = temp
                value = channel_info.get(key)
                if key == "ant_type":
                    value = ant_type_mapping.get(value)  
                value /= norm # normalizing
                for i in range(shape[0]):
                    additional_elements[i, int(channel_id) - 1, idx] = value

        
        return additional_elements

    def add_xyz_information(self, data):
        # Ensure `data` has the expected shape
        assert data.shape[-1] == 512, f"Unexpected last dimension: {data.shape[-1]}"
    
        reshaped_data = data.squeeze(1)  # Reshape [batch, 1, 5, 512] to [batch, 5, 512]
        additional_elements = self.assign_value(reshaped_data.shape) # [batch, 5, 8 ]

        reshaped_data = torch.cat((reshaped_data, additional_elements), dim=-1) # [batch, 5, 520 ]
        reshaped_data = reshaped_data.unsqueeze(1) # Reshape [batch, 5, 520] to [batch, 1, 5, 520] 
        
        # Verify new shape
        assert reshaped_data.shape[-1] == 520, f"Unexpected new shape: {reshaped_data.shape}"

        return reshaped_data
        

    def get_data(self,file_idx):
        # Load data from file

        data, direction, energy , flavor = self.load_file(self.file_ids[file_idx],hyperparameter.norm)

        # randomly choose the points in a file 
        idx = np.random.choice(direction.shape[1], size=self.points, replace=False)

        data = np.swapaxes(data,1,3)
        data = np.swapaxes(data,2,3)

        direction = direction[:,idx]

        data = data[idx,:]
        data = torch.from_numpy(data) # shape [10000, 1, 5, 512])
        # Conditionally add XYZ information
        if self.add_xyz:
            data = self.add_xyz_information(data) # shape [10000, 1, 5, 520])
            

        direction = torch.from_numpy(direction)

        energy = np.expand_dims(energy,1)
        energy = energy[idx, :]
        energy = torch.from_numpy(energy)

        flavor = np.expand_dims(flavor,1)
        flavor = flavor[idx, :]
        flavor = torch.from_numpy(flavor)
        return data, direction, energy, flavor

    # Loading data and label files
    def load_file(self, i_file, norm=1e-6):

        if self.data_type == '_training':
            data_location = hyperparameter.data_location_training

        elif self.data_type == '_validating':
            data_location = hyperparameter.data_location_validating

        elif self.data_type == '_testing':
            data_location = hyperparameter.data_location_testing

        data = np.load(os.path.join(data_location, f"{hyperparameter.data_name}{hyperparameter.interaction}{self.data_type}{'_10k_'}{i_file}_.npy"), allow_pickle=True)
        data = data[:, :, :, np.newaxis]
        # ---- data_processor
        labels_tmp = np.load(os.path.join(data_location, f"{hyperparameter.data_name}{hyperparameter.interaction}{self.data_type}{'_10k_'}{i_file}_labels.npy"), allow_pickle=True)
        azimuths = np.array(labels_tmp.item().get('azimuths'))
        zeniths = np.array(labels_tmp.item().get('zeniths'))

        # check for nans and remove thempython 
        idx = ~(np.isnan(data))
        idx = np.all(idx, axis=1)
        idx = np.all(idx, axis=1)
        idx = np.all(idx, axis=1)
        data = data[idx, :, :, :]
        
        data /= norm

        if hyperparameter.coordinates == 'cartesian':
            direction = np.array([np.sin(zeniths)*np.cos(azimuths), np.sin(zeniths)*np.sin(azimuths), np.cos(zeniths)])
            direction = direction[:, idx]
            return data, direction

        elif hyperparameter.coordinates == 'spherical':
            angle = np.array([zeniths, azimuths])
            angle = angle[:, idx]

            neutrino_energy_data = np.array(labels_tmp.item().get("energies"))
            inelasticity_data = np.array(labels_tmp.item().get("inelasticity"))
            interaction_type_data = np.array(labels_tmp.item().get("interaction_type"))

            inelastic_energy = neutrino_energy_data * inelasticity_data
            mask_of_types = interaction_type_data == 'cc'
            shower_energy_data = np.where(mask_of_types, neutrino_energy_data, inelastic_energy)
            shower_energy_data = shower_energy_data[idx]
            shower_energy_log10 = np.log10(shower_energy_data)
            labels_flavor = np.where(mask_of_types, 1, 0)
            return data, angle, shower_energy_log10, labels_flavor


def data_loader_tester():
    # Data configuration
    data_type = 'testing'  # Options: 'training', 'validating', 'testing'
    label_type = 'nc'      # Options: 'cc', 'nc_cc', 'nc'
    batch_size = hyperparameter.batchSize
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