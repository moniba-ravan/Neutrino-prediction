import numpy as np

"""
The hyperparameters make it possible to alter and change the tunable parameters in a specific model class.
After training the hyperparameters are saved in results/RunXXX.
"""

nc = False
cc = True
nccc = False

if nc:

    n_files_train = 90
    n_files_val = 10
    n_files_test = 5

    #datafile and path name NC
    data_location_training = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/nc/training'
    data_location_validating = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/nc/validating'
    data_location_testing = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/nc/testing'

    interaction = 'nc'

elif cc:

    n_files_train = 90
    n_files_val = 10
    n_files_test = 5

    #datafile and path name CC
    data_location_training = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/cc/training'
    data_location_validating = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/cc/validating'
    data_location_testing = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/cc/testing'

    interaction = 'cc'

elif nccc:

    n_files_train = 180
    n_files_val = 20
    n_files_test = 10

    #datafile and path name NCCC
    data_location_training = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/nc_cc/training'
    data_location_validating = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/nc_cc/validating'
    data_location_testing = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/nc_cc/testing'

    interaction = 'nccc'



n_events_per_file = 10
data_name = 'shallow_baseline_'     #deep_baseline_10k_00000X.npy

filter_path = '/mnt/hdd/nheyer/data/filters/500MHz_filter.npy'

coordinates = 'spherical'
auto_opt = False

#events per data set
train_data_points = n_events_per_file * n_files_train
val_data_points = n_events_per_file * n_files_val
test_data_points = n_events_per_file * n_files_test 

#training parameters
epochs = 2
model_name = "Split"
learning_rate = 0.0005
es_patience = 10
rd_patience = 4
es_min_delta = 0.0001 # Old value: es_min_delta = 0.0001
norm = 1e-6
batchSize = 128
min_lr = 0.000001

worker_num = 0

#model parameters
conv2D_filter_size = 16
pooling_size = 4
amount_Conv2D_layers_per_block = 3 
amount_Conv2D_blocks = 4
conv2D_filter_amount = 32
batch_eps = 0.001
opt_eps = 1e-7
sch_factor = 0.2
momentum = 0.99
padding = 'same'

conv1_layers = 32
conv2_layers = 64
conv3_layers = 128
conv4_layers = 256

output_layer = 1

cond_input = 1000
mlp_layers = '1000'

numb_of_classes = 2
skip_connections = False
