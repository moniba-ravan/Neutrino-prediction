import data_loader
import hyperparameter
import model
import os
import sys
import argparse
import shutil
import numpy as np
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.callbacks import StochasticWeightAveraging
import warnings
warnings.filterwarnings("ignore")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
torch.cuda.empty_cache()
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

#INFORMATION:
#print(train.get_data(0)[0].shape)  #torch.Size([128, 1, 5, 512])

train_loader = DataLoader(train, batch_size=hyperparameter.batchSize, shuffle=False, num_workers=hyperparameter.worker_num)
val_loader = DataLoader(val, batch_size=hyperparameter.batchSize, shuffle=False, num_workers=hyperparameter.worker_num)

#---------load the model

#This is where we load our new model:
model = model.ViTLightningModule()

model.float()
model.pdf_energy.double()
model.pdf_direction.double()

#---------define callbacks
mc = ModelCheckpoint(dirpath=saved_model_dir, filename= 'model_checkpoint', 
    monitor='val_total_loss', verbose=1, save_top_k=3)

es = EarlyStopping("val_total_loss", patience=hyperparameter.es_patience, min_delta=hyperparameter.es_min_delta, verbose=1)

swa = StochasticWeightAveraging(swa_lrs=1e-2)

callbacks = [es, mc, swa]
tb_logger = TensorBoardLogger(saved_model_dir, name="tb_logger", version='version1_')

trainer = pl.Trainer(
    devices=1, 
    callbacks = callbacks, 
    max_epochs = hyperparameter.epochs,
    log_every_n_steps=1,
    logger = tb_logger,
    precision = 32
    )

cont_counter = 0
max_counter = 100

best_checkpoint_path = None
#best_checkpoint_path = '/mnt/hdd/nheyer/gen2_shallow_reco/reconstruction/mix_model/results/Run008/model/model_checkpoint.ckpt'

while cont_counter < max_counter:

    try:
        #---------training
        trainer = pl.Trainer(
            devices=1, 
            callbacks = callbacks, 
            max_epochs = hyperparameter.epochs,
            log_every_n_steps=1,
            logger = tb_logger,
            precision = 32
            )

        trainer.fit(model, train_loader, val_loader, ckpt_path = best_checkpoint_path)
        break

    except Exception as error:
        cont_counter += 1
        print('error: ', error)
        print('Unexpected error, starting continuation = ', cont_counter)

        best_checkpoint_path = saved_model_dir + '/version1_model_checkpoint.ckpt'

        if not os.path.isfile(best_checkpoint_path):
            
            print('starting from the beginning, cont= ', cont_counter)

            best_checkpoint_path = None

        trainer.fit(model, train_loader, val_loader, ckpt_path = best_checkpoint_path)

#---------save model  
torch.save(model.state_dict(), saved_model_dir + '/model_' + args.Run_number + '.pt')
