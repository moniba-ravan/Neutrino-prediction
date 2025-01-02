import torch

#import data_gen
import numpy as np
import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
#import lightning.pytorch as pl
#import models
from tqdm import tqdm
import os
from pytorch_lightning.callbacks import ModelCheckpoint
import jammy_flows
import matplotlib.pyplot as plt

import importlib.util
import sys



parser = argparse.ArgumentParser()

parser.add_argument("--Run_number", type=str, default='RunXXX', help="Name a new Run to be saved")
parser.add_argument("--version", type=str, default='v1', help="Name a new Run to be saved")

parser.add_argument("--test_batchSize", type=int, default=16, help="test batch size")

args = parser.parse_args()

base_dir = '/mnt/hdd/moniba/Neutrino-prediction' 
model_dir = base_dir + '/results/' + args.Run_number + '/model/'
model_file = 'model_checkpoint-' + args.version + '.ckpt'


# load hyperparemeters
hyper_file = base_dir + '/results/' + args.Run_number + '/hyperparameter_' + args.Run_number + '.py'
print(hyper_file)
spec = importlib.util.spec_from_file_location('hyperparameter', hyper_file)
hyperparameter = importlib.util.module_from_spec(spec)
sys.modules['hyperparameter'] = hyperparameter
spec.loader.exec_module(hyperparameter)

# load data_loader
spec = importlib.util.spec_from_file_location('data_loader_gen', base_dir + '/results/' + args.Run_number + '/data_loader_' + args.Run_number + '.py')
data_loader_gen = importlib.util.module_from_spec(spec)
sys.modules['data_loader_gen'] = data_loader_gen
spec.loader.exec_module(data_loader_gen)



# load model architecture
model_arch_path = f'results/{args.Run_number}/{hyperparameter.model}_{args.Run_number}.py'
spec = importlib.util.spec_from_file_location('model_arch', model_arch_path)
model_arch = importlib.util.module_from_spec(spec)
sys.modules['model_arch'] = model_arch
spec.loader.exec_module(model_arch)

list_of_file_ids_test = np.arange(hyperparameter.n_files_test, dtype=int)

test_dataset = data_loader_gen.Prepare_Dataset(file_ids=list_of_file_ids_test,points = hyperparameter.n_events_per_file, data_type = '_testing', worker_num=hyperparameter.worker_num)
#test_dataset = data_loader_gen.Prepare_Dataset(file_ids=list_of_file_ids_test,points = hyperparameter.n_events_per_file, data_type = '_training', worker_num=hyperparameter.worker_num)
test_loader = DataLoader(test_dataset, batch_size=args.test_batchSize, shuffle=False, num_workers=0)


device = torch.device(f"cuda:0")

model = model_arch.get_model()  

print(torch.load(model_dir + model_file)['epoch'])
#quit()

model.float()
model.load_state_dict(torch.load(model_dir + model_file)['state_dict'])
model.pdf_energy.double()
model.pdf_direction.double()
model.eval()



save_path = base_dir + '/results/' + args.Run_number + '/full_y_pred/'
if(not os.path.exists(save_path)):
    os.makedirs(save_path)

total_batch_num = len(test_loader)

results_dict = {}
results_dict['true_zenith'] = np.zeros([total_batch_num * args.test_batchSize])
results_dict['true_azimuth'] = np.zeros([total_batch_num * args.test_batchSize])

results_dict['pred_mean_zenith'] = np.zeros([total_batch_num * args.test_batchSize])
results_dict['pred_mean_azimuth'] = np.zeros([total_batch_num * args.test_batchSize])

results_dict['pred_max_zenith'] = np.zeros([total_batch_num * args.test_batchSize])
results_dict['pred_max_azimuth'] = np.zeros([total_batch_num * args.test_batchSize])

results_dict['true_energy'] = np.zeros([total_batch_num * args.test_batchSize])
results_dict['pred_energy'] = np.zeros([total_batch_num * args.test_batchSize])

results_dict['true_flavor'] = np.zeros([total_batch_num * args.test_batchSize])
results_dict['pred_flavor'] = np.zeros([total_batch_num * args.test_batchSize])

results_dict['pred_area_68'] = np.zeros([total_batch_num * args.test_batchSize])
results_dict['pred_area_50'] = np.zeros([total_batch_num * args.test_batchSize])
results_dict['pred_energy_std'] = np.zeros([total_batch_num * args.test_batchSize])

results_dict['cov_energy_approx'] = np.zeros([total_batch_num * args.test_batchSize])
results_dict['cov_energy_exact'] = np.zeros([total_batch_num * args.test_batchSize])
results_dict['cov_direction_approx'] = np.zeros([total_batch_num * args.test_batchSize])
results_dict['cov_direction_exact'] = np.zeros([total_batch_num * args.test_batchSize])

results_dict['pred_direction_entropy'] = np.zeros([total_batch_num * args.test_batchSize])
results_dict['pred_kl_diff'] = np.zeros([total_batch_num * args.test_batchSize])
results_dict['pred_kl_diff_inv'] = np.zeros([total_batch_num * args.test_batchSize])

#results_dict['cov_direction_exact'] = np.zeros([total_batch_num * args.test_batchSize])
#results_dict['cov_direction_exact'] = np.zeros([total_batch_num * args.test_batchSize])
#results_dict['cov_direction_exact'] = np.zeros([total_batch_num * args.test_batchSize])
#results_dict['cov_direction_exact'] = np.zeros([total_batch_num * args.test_batchSize])
#results_dict['cov_direction_exact'] = np.zeros([total_batch_num * args.test_batchSize])
#results_dict['cov_direction_exact'] = np.zeros([total_batch_num * args.test_batchSize])


#samplesize()
n = 10000
lll = 0

with torch.no_grad():
    for count, test_quant in enumerate(tqdm(test_loader)):

        first_item = count * args.test_batchSize
        last_item = count * args.test_batchSize + args.test_batchSize 

        x_test, true_direction, true_energy, true_flavor = test_quant

        results_dict['true_flavor'][first_item : last_item] = true_flavor.detach().cpu().numpy().squeeze()
        results_dict['true_energy'][first_item : last_item] = true_energy.detach().cpu().numpy().squeeze()
        results_dict['true_zenith'][first_item : last_item] = true_direction[:, 0].detach().cpu().numpy().squeeze()
        results_dict['true_azimuth'][first_item : last_item] = true_direction[:, 1].detach().cpu().numpy().squeeze()

        conv_out = model.forward(x_test)

        flavor_pred = model.classifier(conv_out).squeeze()
        results_dict['pred_flavor'][first_item : last_item] = flavor_pred.detach().cpu().numpy().squeeze()


        moments_dict_energy = model.pdf_energy.marginal_moments(conditional_input=conv_out.to(torch.double).to(device), calc_kl_diff_and_entropic_quantities=True)
        coverage_dict_energy = model.pdf_energy.coverage_and_or_pdf_scan(labels = true_energy.to(torch.double).to(device), conditional_input=conv_out.to(torch.double).to(device), exact_coverage_calculation=True)


        results_dict['pred_energy'][first_item : last_item] = moments_dict_energy['mean_0'].squeeze()
        results_dict['pred_energy_std'][first_item : last_item] = moments_dict_energy['varlike_0'].squeeze()
        results_dict['cov_energy_approx'] [first_item : last_item] = coverage_dict_energy['approx_cov_values'].squeeze()
        results_dict['cov_energy_exact'] [first_item : last_item] = coverage_dict_energy['real_cov_values'].squeeze()

        moments_dict_direction = model.pdf_direction.marginal_moments(conditional_input=conv_out.to(torch.double).to(device), calc_kl_diff_and_entropic_quantities=True)
        coverage_dict_direction = model.pdf_direction.coverage_and_or_pdf_scan(labels = true_direction.to(torch.double).to(device), conditional_input=conv_out.to(torch.double).to(device), exact_coverage_calculation=True, save_pdf_scan=True, calculate_MAP=True)
        
        areas_50_list = []
        areas_68_list = []
        

        for indx, event in enumerate(coverage_dict_direction['pdf_scan_positions']):

            batch_positions = coverage_dict_direction['pdf_scan_positions'][indx]
            batch_sizes = coverage_dict_direction['pdf_scan_volume_sizes'][indx]
            batch_evals = coverage_dict_direction['pdf_scan_log_evals'][indx]

            xyz_positions_pred, _ = model.pdf_direction.layer_list[0][0].spherical_to_eucl_embedding(torch.tensor(batch_positions), 0.0)
            xyz_positions_pred = xyz_positions_pred.numpy()

            large_to_small_probs_mask_direction = np.argsort(batch_evals)[::-1]
            sorted_pos_direction = xyz_positions_pred[large_to_small_probs_mask_direction]
            sorted_areas = batch_sizes[large_to_small_probs_mask_direction]
            sorted_evals_direction = sorted_areas * np.exp(batch_evals[large_to_small_probs_mask_direction])

            cumsum_direction = np.cumsum(sorted_evals_direction)

            pixel_68 = np.argmin(abs(0.6827 - cumsum_direction))
            area_68 = np.sum(sorted_areas[:pixel_68])
            areas_68_list.append(area_68)

            pixel_50 = np.argmin(abs(0.5 - cumsum_direction))
            area_50 = np.sum(sorted_areas[:pixel_50])
            areas_50_list.append(area_50)

        results_dict['pred_mean_zenith'][first_item : last_item] = moments_dict_direction['mean_0_angles'][:, 0].squeeze()
        results_dict['pred_mean_azimuth'][first_item : last_item] = moments_dict_direction['mean_0_angles'][:, 1].squeeze()
        results_dict['pred_max_zenith'][first_item : last_item] = moments_dict_direction['argmax_0_angles'][:, 0].squeeze()
        results_dict['pred_max_azimuth'][first_item : last_item] = moments_dict_direction['argmax_0_angles'][:, 1].squeeze()
        results_dict['pred_area_68'][first_item : last_item] = areas_68_list
        results_dict['pred_area_50'][first_item : last_item] = areas_50_list
        results_dict['cov_direction_approx'][first_item : last_item] = coverage_dict_direction['approx_cov_values'].squeeze()
        results_dict['cov_direction_exact'][first_item : last_item] = coverage_dict_direction['real_cov_values'].squeeze()
        results_dict['pred_direction_entropy'] = moments_dict_direction['approx_entropy_0'].squeeze()
        results_dict['pred_kl_diff'] = moments_dict_direction['kl_diff_exact_approx_0'].squeeze()
        results_dict['pred_kl_diff_inv'] = moments_dict_direction['kl_diff_approx_exact_0'].squeeze()

        #print(moments_dict_direction.keys())

        lll += 1
        
        if lll == 600:
            np.save(save_path + 'results-' + args.version + '.npy', results_dict)

np.save(save_path + 'results-' + args.version + '.npy', results_dict)











