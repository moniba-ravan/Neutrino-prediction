import numpy as np

"""
The hyperparameters make it possible to alter and change the tunable parameters in a specific model class.
After training the hyperparameters are saved in results/RunXXX.
"""

#I use cc because it has less files than nccc so it is faster.
nc = False
cc = False
nccc = True


#DEFAULT SETTINGS FOR TESTING:
# nccc, 18 train and 2 val files, 10000 events per file
# epochs = 10

#Settings for nc1, cc1 & nccc1: 6 train, 2 val, 5 test, 10000 events, 10 epochs

if nc:

    n_files_train = 6
    n_files_val = 2
    n_files_test = 5

    #datafile and path name NC
    data_location_training = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/nc/training'
    data_location_validating = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/nc/validating'
    data_location_testing = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/nc/testing'

    interaction = 'nc'

elif cc:

    n_files_train = 6      #90
    n_files_val = 2        #10
    n_files_test = 5

    #datafile and path name CC
    data_location_training = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/cc/training'
    data_location_validating = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/cc/validating'
    data_location_testing = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/cc/testing'

    interaction = 'cc'

elif nccc:

    n_files_train = 180    #180
    n_files_val = 20      #20
    n_files_test = 10     #10

    #datafile and path name NCCC
    data_location_training = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/nc_cc/training'
    data_location_validating = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/nc_cc/validating'
    data_location_testing = '/mnt/hdd/nheyer/data/gen2_shallow_baseline/nc_cc/testing'

    interaction = 'nccc'


#Here I make it faster by using less events per file.
n_events_per_file = 10000
data_name = 'shallow_baseline_'     #deep_baseline_10k_00000X.npy

filter_path = '/mnt/hdd/nheyer/data/filters/500MHz_filter.npy'

coordinates = 'spherical'
auto_opt = False

#events per data set
train_data_points = n_events_per_file * n_files_train
val_data_points = n_events_per_file * n_files_val
test_data_points = n_events_per_file * n_files_test 

#training parameters
epochs = 250    #250
model_name = "Trans"
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

cond_input = 512       #Set to 512 for ViT & 1000 for ResNet
mlp_layers = '512'  

numb_of_classes = 2
skip_connections = False



# antennas file info location
detector_file_location = '/mnt/hdd/nheyer/dl_playground/sim_file/detector.json'
add_xyz = True


model = "vit"  # Options: "resnet34", "vit", "transformer"

