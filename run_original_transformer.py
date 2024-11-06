import data_loader
import hyperparameter
import model
import torch
import pytorch_lightning as pl
from model import TransformerModel  # Ensure this points to your `TransformerModel`

import os
import sys
import argparse
import shutil
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import StochasticWeightAveraging
import jammy_flows 
"""
Calls hyperparameter, model and data_loader. Trains the model on the specified training data and saves the trained model. 
"""

#---------create file structure
parser = argparse.ArgumentParser()
parser.add_argument("--Run_number", type=str, default='RunXXX', help="Name a new Run to be saved")
args = parser.parse_args()

saved_dir = 'results/' + args.Run_number
saved_model_dir = 'results/' + args.Run_number + '/model'
saved_plots_dir = 'results/' + args.Run_number + '/plots'

if saved_dir == 'results/RunXXX': 
    if not os.path.exists(saved_plots_dir):
        os.makedirs(saved_plots_dir)
    if not os.path.exists(saved_model_dir):
        os.mkdir(saved_model_dir)
else:
    assert(not os.path.exists(saved_dir))
    os.makedirs(saved_plots_dir)
    os.mkdir(saved_model_dir)

src_hyper = 'hyperparameter.py'
dst_hyper = 'results/' + args.Run_number + '/hyperparameter_' + args.Run_number + '.py'
shutil.copy(src_hyper, dst_hyper)

src_model= 'model.py'
dst_model = 'results/' + args.Run_number + '/model_' + args.Run_number + '.py'
shutil.copy(src_model, dst_model)

src_run = 'run.py'
dst_run = 'results/' + args.Run_number + '/run_' + args.Run_number + '.py'
shutil.copy(src_run, dst_run)

src_data = 'data_loader.py'
dst_data = 'results/' + args.Run_number + '/data_loader_' + args.Run_number + '.py'
shutil.copy(src_data, dst_data)

#---------load the data

list_of_file_ids_train = np.arange(hyperparameter.n_files_train, dtype=int)
list_of_file_ids_val = np.arange(hyperparameter.n_files_val, dtype=int)

train = data_loader.Prepare_Dataset(file_ids=list_of_file_ids_train, points = hyperparameter.n_events_per_file, data_type = '_training', worker_num=hyperparameter.worker_num)
val = data_loader.Prepare_Dataset(file_ids=list_of_file_ids_val, points = hyperparameter.n_events_per_file, data_type = '_validating', worker_num=hyperparameter.worker_num)

train_loader = DataLoader(train, batch_size=hyperparameter.batchSize, shuffle=False, num_workers=hyperparameter.worker_num)
val_loader = DataLoader(val, batch_size=hyperparameter.batchSize, shuffle=False, num_workers=hyperparameter.worker_num)

#---------load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model with updated arguments
model = TransformerModel(embed_dim=512, num_heads=8, num_encoders=6, dropout=0.1).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=1e-4, weight_decay=0)

#I'm not even sure if I put this function in the correct spot or if it should be in the model class, let's discuss sometime
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device):
    ii = 0   #To keep count of how many training steps we take
    model = model.to(device)  #I put this as they always do but GPU did not really work for me so I use CPU
    for epoch in range(epochs):
        model.train()  #Set model to training mode
        total_train_loss = 0
        
        #Training step
        for batch_idx, (inputs, targets, *_) in enumerate(train_loader):
            ii += 1
            print(ii)    #Count training step

            inputs, targets = inputs.to(device), targets.to(device)
            #I use the following line to remove an unwanted dimension because target was (128, 2), this might be sus to do idk.
            targets = targets.argmax(dim=1)
            optimizer.zero_grad()  #Clear gradients from the last iteration
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Update training loss
            total_train_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}")
        
        # Validation step
        model.eval()  # Set model to evaluation mode
        total_val_loss = 0
        correct = 0
        with torch.no_grad():  # No need to calculate gradients for validation
            for inputs, targets, *_ in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                #Same sus thing as above here.
                targets = targets.argmax(dim=1)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update validation loss
                total_val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == targets).sum().item()
        
        # Calculate average validation loss and accuracy for the epoch
        avg_val_loss = total_val_loss / len(val_loader)
        val_accuracy = correct / len(val_loader.dataset)
        
        print(f"Epoch [{epoch+1}/{epochs}], Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
    print("Training complete.")

#Use the training function
train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=hyperparameter.epochs,
    device=device
)