import numpy as np

"""
Hyperparameters for the model training.
This script defines tunable parameters for the model, dataset, and training process.
"""

# ==============================
# Model Configuration
# ==============================

model = "vit_model"  # Options: "resnet34", "vit_model", "transformer_model"
base_dir = '/mnt/hdd/moniba/Neutrino-prediction'


# ==============================
# Detector and Data Configuration
# ==============================

detector_file_location = '/mnt/hdd/nheyer/dl_playground/sim_file/detector.json'
add_xyz = False

# ==============================
# Interaction Types: Select the interaction type (nc, cc, nccc)
# ==============================
nc = False
cc = False
nccc = True

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


# ==============================
# Dataset Configuration
# ==============================

n_events_per_file = 10000 # Number of events per file
data_name = 'shallow_baseline_'     #deep_baseline_10k_00000X.npy

# Filter path for preprocessing
filter_path = '/mnt/hdd/nheyer/data/filters/500MHz_filter.npy'

# Coordinate system for dataset
coordinates = 'spherical'
auto_opt = False

# Total events for each dataset
train_data_points = n_events_per_file * n_files_train
val_data_points = n_events_per_file * n_files_val
test_data_points = n_events_per_file * n_files_test 


# ==============================
# Training Configuration
# ==============================

epochs = 250    #250
model_name = "Trans"
learning_rate = 0.0005
es_patience = 10
rd_patience = 4
es_min_delta = 0.0001 # Old value: es_min_delta = 0.0001
norm = 1e-6
batchSize = 128
min_lr = 0.000001

# Number of workers for data loading
worker_num = 0

# ==============================
# Model Architecture Configuration
# ==============================

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

# Number of filters for each convolutional layer
conv1_layers = 32
conv2_layers = 64
conv3_layers = 128
conv4_layers = 256

output_layer = 1

# ==============================
# Model-Specific Configuration
# ==============================
cond_input = 512       #Set to 512 for ViT & 1000 for ResNet
mlp_layers = '512'  

numb_of_classes = 2
skip_connections = False
