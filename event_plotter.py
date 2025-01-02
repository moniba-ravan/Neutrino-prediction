import torch

#import data_gen
import numpy as np
import argparse
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from tqdm import tqdm
import os
from pytorch_lightning.callbacks import ModelCheckpoint

import jammy_flows.helper_fns as helper_fns
from astropy.visualization.wcsaxes import WCSAxes
import matplotlib.gridspec as gridspec
from mhealpy.plot.axes import HealpyAxes
from matplotlib.projections import register_projection
from astropy.visualization.wcsaxes.frame import EllipticalFrame
from matplotlib.transforms import Bbox
from matplotlib.colors import LogNorm
from numpy import ma
from astropy import units as u
from torch import nn



import pylab
import jammy_flows

import importlib.util
import sys

sys.path.insert(1, '..')

import data_gen
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
from astrotools import healpytools as hpt
import meander 
import data_gen as data_gen

import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as pe
from matplotlib import pylab
from matplotlib.lines import Line2D
import torch.nn.functional as F

#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]




parser = argparse.ArgumentParser()

parser.add_argument("--Run_number", type=str, default='RunXXX', help="Name a new Run to be saved")
parser.add_argument("--version", type=str, default='RunXXX', help="Name a new Run to be saved")
parser.add_argument("--test_batchSize", type=int, default=16, help="test batch size")
args = parser.parse_args()

model_dir = '/mnt/hdd/nheyer/gen2_shallow_reco/reconstruction/mix_split/results/' + args.Run_number + '/model/'
model_file = 'model_checkpoint.ckpt'

spec = importlib.util.spec_from_file_location('hyperparameters', '/mnt/hdd/nheyer/gen2_shallow_reco/reconstruction/mix_split/results/' + args.Run_number + '/hyperparameters_' + args.Run_number + '.py')
hyperparameters = importlib.util.module_from_spec(spec)
sys.modules['hyperparameters'] = hyperparameters
spec.loader.exec_module(hyperparameters)

spec = importlib.util.spec_from_file_location('data_gen', '/mnt/hdd/nheyer/gen2_shallow_reco/reconstruction/mix_split/results/' + args.Run_number + '/data_' + args.Run_number + '.py')
data_gen = importlib.util.module_from_spec(spec)
sys.modules['data_gen'] = data_gen
spec.loader.exec_module(data_gen)

spec = importlib.util.spec_from_file_location('models', '/mnt/hdd/nheyer/gen2_shallow_reco/reconstruction/mix_split/results/' + args.Run_number + '/models_' + args.Run_number + '.py')
models = importlib.util.module_from_spec(spec)
sys.modules['models'] = models
spec.loader.exec_module(models)

list_of_file_ids_test = np.arange(hyperparameters.n_files_test, dtype=int)
test_dataset = data_gen.Prepare_Dataset_direction(file_ids=list_of_file_ids_test,points = hyperparameters.n_events_per_file, data_type = '_testing', worker_num=hyperparameters.worker_num)
test_loader = DataLoader(test_dataset, batch_size=args.test_batchSize, shuffle=False, num_workers=0)

device = torch.device(f"cuda:0")
model = models.Split(loss_function = hyperparameters.loss_function)
model.float()
model.load_state_dict(torch.load(model_dir + model_file)['state_dict'])
model.pdf_energy.double()
model.pdf_direction.double()
model.eval()




def autolabel(rects, count_labels, platform):
    # attach some text labels
        for ii,rect in enumerate(rects):

            width =  rect.get_width()

            height = rect.get_height()

            yloc1=rect.get_y() + height /2.0
            yloc2=rect.get_y() + height /2.0
            if (width <= 40):
                # Shift the text to the right side of the right edge
                xloc1 = width + 1
                xloc2 = 70
                #yloc2=yloc2+0.3
                # Black against white background
                clr = 'black'
                align = 'left'
            else:
                # Shift the text to the left side of the right edge
                xloc1 = 0.98*width
                xloc2 = 5
                # White on blue
                clr = 'white'
                align = 'right'
            yloc1=rect.get_y() + height /2.0

            flavor_prob.text(xloc1,yloc1, '%s'% (count_labels[ii]),horizontalalignment=align,
                             verticalalignment='center',color=clr,weight='bold',
                             clip_on=True)
            flavor_prob.text(xloc2,yloc2, '%s'% (platform[ii]),horizontalalignment='left',
                             verticalalignment='center',color=clr,weight='bold',
                             clip_on=True)


conversion = 3282.80635
samples_per_event = 10000

with torch.no_grad():
    for count, test_quant in enumerate(tqdm(test_loader)):

        x_test, true_direction, true_energy, true_flavor = test_quant



        conv_out = model.forward(x_test)

        #for every sample make a prediction for the flavor type and reshape it back for every event
        flavor_pred = model.classifier(conv_out).squeeze()


        moments_dict_energy = model.pdf_energy.marginal_moments(conditional_input=conv_out.to(torch.double).to(device), calc_kl_diff_and_entropic_quantities=True)
        coverage_dict_energy = model.pdf_energy.coverage_and_or_pdf_scan(labels = true_energy.to(torch.double).to(device), conditional_input=conv_out.to(torch.double).to(device), exact_coverage_calculation=True)


        pred_energy = moments_dict_energy['mean_0'].squeeze()
        pred_energy_std = moments_dict_energy['varlike_0'].squeeze()
        cov_energy_approx = coverage_dict_energy['approx_cov_values'].squeeze()
        cov_energy_exact = coverage_dict_energy['real_cov_values'].squeeze()

        moments_dict_direction = model.pdf_direction.marginal_moments(conditional_input=conv_out.to(torch.double).to(device), calc_kl_diff_and_entropic_quantities=True)
        coverage_dict_direction = model.pdf_direction.coverage_and_or_pdf_scan(labels = true_direction.to(torch.double).to(device), conditional_input=conv_out.to(torch.double).to(device), exact_coverage_calculation=True, save_pdf_scan=True, calculate_MAP=True)

        true_flavor = true_flavor.squeeze()
        true_energy = true_energy.squeeze()
        true_zenith = true_direction[:, 0].squeeze()
        true_azimuth = true_direction[:, 1].squeeze()

        pred_mean_zenith = moments_dict_direction['mean_0_angles'][:, 0].squeeze()
        pred_mean_azimuth = moments_dict_direction['mean_0_angles'][:, 1].squeeze()


        for event in range(len(true_energy)):
            
            
            
            if true_zenith[event].item() > np.pi/2:

                print('true zenith', np.rad2deg(true_zenith[event].item()))
                print('cut zenith', np.rad2deg(np.pi/2))
                continue
            if true_zenith[event].item() < 5*np.pi/12:
                continue
            if true_azimuth[event].item() < 4*np.pi/3 - 0.05:
                print('true azimuth', np.rad2deg(true_azimuth[event].item()))
                print('cut down azimuth', np.rad2deg(4*np.pi/3 - 0.05))
                
                continue
            if true_azimuth[event].item() > 4*np.pi/3 + 0.05:
                print('true azimuth', np.rad2deg(true_azimuth[event].item()))
                print('cut up azimuth', np.rad2deg(4*np.pi/3 + 0.05))
                continue
            
            if np.abs(true_energy[event].item() - pred_energy[event].item()) > 0.5:
                print('true azimuth', np.rad2deg(true_azimuth[event].item()))
                print('cut up azimuth', np.rad2deg(4*np.pi/3 + 0.05))
                continue 

            if np.abs(true_energy[event].item() - pred_energy[event].item()) > 0.5:
                print('true azimuth', np.rad2deg(true_azimuth[event].item()))
                print('cut up azimuth', np.rad2deg(4*np.pi/3 + 0.05))
                continue 
            
            

            #energy evaluation

            data_summary_repeated=conv_out.to(torch.double).to(device)[event:event+1].repeat_interleave(samples_per_event, dim=0)
            samples,_,log_pdf_at_samples,_=model.pdf_energy.sample(conditional_input=data_summary_repeated, samplesize=samples_per_event)
            log_pdf_at_samples=log_pdf_at_samples.cpu().numpy()

            
            _, density_bounds,_=jammy_flows.helper_fns.grid_functions.obtain_bins_and_visualization_regions(samples, model.pdf_energy, percentiles=[0.5,99.5])
            npts_per_dim=int((samples_per_event)**(1.0/float(model.pdf_energy.total_target_dim)))

            evalpositions, log_evals, bin_volumes, _, _= jammy_flows.helper_fns.grid_functions.get_pdf_on_grid(density_bounds,
                                                                                                npts_per_dim,
                                                                                                model.pdf_energy,
                                                                                                conditional_input= data_summary_repeated[:1],
                                                                                                )
                                                                                                #s2_norm="standard",
                                                                                                #s2_rotate_to_true_value=False,
                                                                                                #true_values=None)


            energy_eval_pos = np.insert(evalpositions.squeeze(), 0, 16.0)
            energy_eval_pos = np.insert(energy_eval_pos, -1, 20.2)
            energy_eval = np.insert(np.exp(log_evals.squeeze()), 0, 0)
            energy_eval = np.insert(energy_eval, -1, 0)
            normed_evals_energy = bin_volumes * energy_eval

            
            sorted_evals_energy = np.sort(normed_evals_energy)[::-1]
            cumsum_energy = np.cumsum(sorted_evals_energy)

            cross_value_arg = np.argmin(abs(0.6827 - cumsum_energy))
            cross_value = sorted_evals_energy[cross_value_arg]

            outside_values = np.where(normed_evals_energy > cross_value, 0, normed_evals_energy)
            inside_values = np.where(normed_evals_energy <= cross_value, 0, normed_evals_energy)

            normed_all = normed_evals_energy / max(normed_evals_energy)
            normed_out = outside_values / max(normed_evals_energy)
            normed_ins = inside_values / max(normed_evals_energy)
            


            cumsum_energy = np.cumsum(normed_evals_energy)

            pixel_energy_16 = np.argmin(abs(0.15865 - cumsum_energy))
            pixel_energy_84 = np.argmin(abs(0.841335 - cumsum_energy))

            #direction eval
            batch_positions = coverage_dict_direction['pdf_scan_positions'][event]
            batch_sizes = coverage_dict_direction['pdf_scan_volume_sizes'][event]
            batch_evals = coverage_dict_direction['pdf_scan_log_evals'][event]

            xyz_positions_pred, _ = model.pdf_direction.layer_list[0][0].spherical_to_eucl_embedding(torch.tensor(batch_positions), 0.0)
            xyz_positions_pred = xyz_positions_pred.numpy()

            large_to_small_probs_mask_direction = np.argsort(batch_evals)[::-1]
            sorted_pos_direction = xyz_positions_pred[large_to_small_probs_mask_direction]
            sorted_areas = batch_sizes[large_to_small_probs_mask_direction]
            sorted_evals_direction = sorted_areas * np.exp(batch_evals[large_to_small_probs_mask_direction])

            cumsum_direction = np.cumsum(sorted_evals_direction)

            pixel_68 = np.argmin(abs(0.6827 - cumsum_direction))
            area_68 = np.sum(sorted_areas[:pixel_68]) * conversion


            if area_68 > 100:
                print('area_68', area_68)
                continue
            
            #inisializing the figure
            fig = plt.figure(figsize=(8, 10), constrained_layout = True)#layout='constrained')

            gs = fig.add_gridspec(4, 2, height_ratios=[4, 4, 3, 1])#, layout='constrained')
            gs = fig.add_gridspec(3, 2, height_ratios=[4, 4, 3])#, layout='constrained')
            direction_full = fig.add_subplot(gs[0, :])
            direction_zoom = fig.add_subplot(gs[1, :])
            energy_pdf = fig.add_subplot(gs[2, 0])
            flavor_prob = fig.add_subplot(gs[2, 1])
            #sum_table = fig.add_subplot(gs[3, :])


            #energy plotting

            energy_pdf.plot(energy_eval_pos, normed_all, label='predicted pdf')
            #energy_pdf.axvline(pred_energy[event], color='k', linestyle='--', label='predicted energy')
            energy_pdf.fill_between(energy_eval_pos, normed_out, alpha = 0.5, color='tab:blue')
            energy_pdf.fill_between(energy_eval_pos, normed_ins, color='tab:orange', alpha = 0.8 ,label='68%')
            #energy_pdf.fill_between(energy_eval_pos[pixel_energy_84:], normed_evals_energy[pixel_energy_84:]/np.max(normed_evals_energy), alpha = 0.5, color='tab:blue')
            #energy_pdf.axvline(true_energy[event], color='r', label='true energy')
            

            #energy_pdf.axvline(energy_eval_pos[pixel_energy_16], color='tab:orange', label='16 percentile')
            #energy_pdf.axvline(energy_eval_pos[pixel_energy_84], color='tab:orange', label='84 percentile')
            
            energy_pdf.set_xlabel("$\log_{10}\:E_{shower}$")
            energy_pdf.set_ylabel("normalized amplitude")
            energy_pdf.set_xlim(16, 20)
            energy_pdf.set_ylim(0)

            energy_pdf.legend()


            #flavor plotting
            flavor_names = [r"$\nu_e$ - NC", r"$\nu_e$ - CC"]
            flavor_names = [r"had.", r"had. + EM"]
            flavor_percentage = [100 * (1 - flavor_pred[event]), 100 * flavor_pred[event]]
            count_labels = [np.round(100 * (1 - flavor_pred[event])).item(), np.round(100 * flavor_pred[event]).item()]

            #if true_flavor[event] == 1:
            #    flavor_prob.fill_between([0, 100], [1.5, 1.5], [0.5, 0.5], alpha = 0.5, color = 'tab:green')
            #if true_flavor[event] == 0:
            #    flavor_prob.fill_between([0, 100], [0.5, 0.5], [-0.5, -0.5], alpha = 0.5, color = 'tab:green')
        
            rects = flavor_prob.barh(flavor_names, flavor_percentage, align='center', height=0.5, color= 'k')
            flavor_prob.set_xlim(0, 100)
            flavor_prob.set_xlabel("Interaction Probability [%]")
            #flavor_prob.plot([0, 100], [0.5, 0.5], color = 'tab:green')
            flavor_prob.set_yticks([])
            autolabel(rects, count_labels, flavor_names)

            #direction plotting
            direction_full = jammy_flows.helper_fns.plotting.spherical.plot_multiresolution_healpy(model.pdf_direction,
                                fig=fig, 
                                ax_to_plot=direction_full,
                                samplesize=10000,
                                conditional_input=conv_out.to(torch.double).to(device)[event], 
                                sub_pdf_index=0,
                                max_entries_per_pixel=5,
                                draw_pixels=True,
                                use_density_if_possible=True,
                                log_scale=False,
                                cbar=True,
                                cbar_kwargs={},
                                graticule=True,
                                graticule_kwargs={},
                                draw_contours=False,
                                contour_probs=[0.68, 0.95],
                                contour_colors=None, # None -> pick colors from color scheme
                                zoom=False,
                                normed= True,
                                zoom_contained_prob_mass=0.97,
                                projection_type="zen_azi", # zen_azi or dec_ra
                                declination_trafo_function=None, # required to transform to dec/ra before plotting 
                                show_grid=False)

            
            text1 = direction_full.text(180, 200, 'predicted pdf', fontsize=15, color = 'w')
            arrow1 = direction_full.arrow(250, 190, 50, -40, width=5, color = 'w')

            #direction zoom plotting
            direction_zoom = jammy_flows.helper_fns.plotting.spherical.plot_multiresolution_healpy(model.pdf_direction,
                                fig=fig, 
                                ax_to_plot=direction_zoom,
                                samplesize=10000,
                                conditional_input=conv_out.to(torch.double).to(device)[event], 
                                sub_pdf_index=0,
                                max_entries_per_pixel=5,
                                draw_pixels=True,
                                use_density_if_possible=True,
                                log_scale=False,
                                cbar=False,
                                cbar_kwargs={},
                                graticule=True,
                                graticule_kwargs={},
                                draw_contours=True,
                                contour_probs=[0.68],
                                contour_colors=['tab:orange'], # None -> pick colors from color scheme
                                zoom=True,
                                zoom_contained_prob_mass=0.99,
                                projection_type="zen_azi", # zen_azi or dec_ra
                                declination_trafo_function=None, # required to transform to dec/ra before plotting 
                                show_grid=False)

            #direction_zoom.proj_plot([true_zenith[event].item()], [true_azimuth[event].item()],c='r',marker='x',label='true direction', zorder=2, s=150, linewidths=4)
            #direction_zoom.proj_plot([pred_mean_zenith[event].item()], [pred_mean_azimuth[event].item()],c='k',marker='x',label='predicted direction', zorder=2, s=150, linewidths=4)
            #direction_zoom.legend(bbox_to_anchor=(0.2, 1.2), bbox_transform=direction_zoom.transAxes)

            """
            #table plotting
            #col_labels = [r'Flavor [$\nu_e \: CC - \%$]',r'Energy$_{log}$ [$eV$]', r'Zenith [$deg$]',r'Azimuth [$deg$]', r"Area [$deg^2$]"]
            col_labels = [r'Flavor [had. + EM %]',r'Energy$_{log}$ [$eV$]', r'Zenith [$deg$]',r'Azimuth [$deg$]', r"Area [$deg^2$]"]

            row_labels = ['predicted']
            table_vals =    [[np.round(100*flavor_pred[event].item(), 2), str(np.round(pred_energy[event], 2)) + r'$\pm$' +  str(np.round(np.sqrt(moments_dict_energy['varlike_0'].squeeze()[event]), 2)), np.round(np.rad2deg(pred_mean_zenith[event].item()), 2), np.round(np.rad2deg(pred_mean_azimuth[event].item()), 2), np.round(area_68, 2)],
                            ]

            sum_table.axis("off")  
            my_table = sum_table.table(cellText=table_vals,
                                #colWidths=[0.2] * 3,
                                rowLabels=row_labels,
                                colLabels=col_labels,
                                #rowColours=row_colors,
                                loc='center right')
            """

            plt.savefig('/mnt/hdd/nheyer/gen2_shallow_reco/reconstruction/mix_split/results/' + args.Run_number + '/plots/' + args.version + '/event_testing.png', dpi=300)
            plt.close()




            #inisializing the figure
            fig = plt.figure(figsize=(8, 10), constrained_layout = True)#layout='constrained')

            gs = fig.add_gridspec(4, 2, height_ratios=[4, 4, 3, 1])#, layout='constrained')
            gs = fig.add_gridspec(3, 2, height_ratios=[4, 4, 3])#, layout='constrained')
            direction_full = fig.add_subplot(gs[0, :])
            direction_zoom = fig.add_subplot(gs[1, :])
            energy_pdf = fig.add_subplot(gs[2, 0])
            flavor_prob = fig.add_subplot(gs[2, 1])
            #sum_table = fig.add_subplot(gs[3, :])


            #energy plotting
            energy_pdf.plot(energy_eval_pos, normed_all, label='predicted pdf')
            energy_pdf.axvline(true_energy[event], color='k', linestyle='--', label='true energy')
            #energy_pdf.axvline(pred_energy[event], color='k', linestyle='--', label='predicted energy')
            energy_pdf.fill_between(energy_eval_pos, normed_out, alpha = 0.5, color='tab:blue')
            energy_pdf.fill_between(energy_eval_pos, normed_ins, color='tab:orange', alpha = 0.8 ,label='68%')


            #energy_pdf.plot(energy_eval_pos, normed_evals_energy/np.max(normed_evals_energy), label='PDF')
            #energy_pdf.axvline(pred_energy[event], color='k', linestyle='--', label='predicted energy')
            #energy_pdf.axvline(true_energy[event], color='r', linestyle=':', label='true energy')
            #energy_pdf.fill_between(energy_eval_pos[:pixel_energy_16], normed_evals_energy[:pixel_energy_16]/np.max(normed_evals_energy), alpha = 0.5, color='tab:blue')
            #energy_pdf.fill_between(energy_eval_pos[pixel_energy_16:pixel_energy_84], normed_evals_energy[pixel_energy_16:pixel_energy_84]/np.max(normed_evals_energy), color='tab:orange', alpha = 0.8 ,label='68%')
            #energy_pdf.fill_between(energy_eval_pos[pixel_energy_84:], normed_evals_energy[pixel_energy_84:]/np.max(normed_evals_energy), alpha = 0.5, color='tab:blue')
            #energy_pdf.axvline(true_energy[event], color='r', label='true energy')
            

            #energy_pdf.axvline(energy_eval_pos[pixel_energy_16], color='tab:orange', label='16 percentile')
            #energy_pdf.axvline(energy_eval_pos[pixel_energy_84], color='tab:orange', label='84 percentile')
            
            energy_pdf.set_xlabel("$\log_{10}\:E_{shower}$")
            energy_pdf.set_ylabel("normalized amplitude")
            energy_pdf.set_xlim(16, 20)
            energy_pdf.set_ylim(0)

            energy_pdf.legend()
            flavor_names = [r"$\nu_e$ - NC", r"$\nu_e$ - CC"]
            flavor_names = [r"had.", r"had. + EM"]
            flavor_percentage = [100 * (1 - flavor_pred[event]), 100 * flavor_pred[event]]
            count_labels = [np.round(100 * (1 - flavor_pred[event])).item(), np.round(100 * flavor_pred[event]).item()]

            #flavor plotting
            if true_flavor[event] == 1:
                flavor_prob.fill_between([0, 100], [1.5, 1.5], [0.5, 0.5], alpha = 0.5, color = 'tab:green')
            if true_flavor[event] == 0:
                flavor_prob.fill_between([0, 100], [0.5, 0.5], [-0.5, -0.5], alpha = 0.5, color = 'tab:green')
        
            rects = flavor_prob.barh(flavor_names, flavor_percentage, align='center', height=0.5, color= 'k')
            flavor_prob.set_xlim(0, 100)
            flavor_prob.set_xlabel("Interaction Probability [%]")
            flavor_prob.plot([0, 100], [0.5, 0.5], color = 'tab:green')
            flavor_prob.set_yticks([])
            autolabel(rects, count_labels, flavor_names)

            #direction plotting
            direction_full = jammy_flows.helper_fns.plotting.spherical.plot_multiresolution_healpy(model.pdf_direction,
                                fig=fig, 
                                ax_to_plot=direction_full,
                                samplesize=10000,
                                conditional_input=conv_out.to(torch.double).to(device)[event], 
                                sub_pdf_index=0,
                                max_entries_per_pixel=5,
                                draw_pixels=True,
                                use_density_if_possible=True,
                                log_scale=False,
                                cbar=True,
                                cbar_kwargs={},
                                graticule=True,
                                graticule_kwargs={},
                                draw_contours=False,
                                contour_probs=[0.68, 0.95],
                                contour_colors=None, # None -> pick colors from color scheme
                                zoom=False,
                                normed= True,
                                zoom_contained_prob_mass=0.97,
                                projection_type="zen_azi", # zen_azi or dec_ra
                                declination_trafo_function=None, # required to transform to dec/ra before plotting 
                                show_grid=False)

            
            text1 = direction_full.text(180, 200, 'predicted pdf', fontsize=15, color = 'w')
            arrow1 = direction_full.arrow(250, 190, 50, -40, width=5, color = 'w')

            #direction zoom plotting
            direction_zoom = jammy_flows.helper_fns.plotting.spherical.plot_multiresolution_healpy(model.pdf_direction,
                                fig=fig, 
                                ax_to_plot=direction_zoom,
                                samplesize=10000,
                                conditional_input=conv_out.to(torch.double).to(device)[event], 
                                sub_pdf_index=0,
                                max_entries_per_pixel=5,
                                draw_pixels=True,
                                use_density_if_possible=True,
                                log_scale=False,
                                cbar=False,
                                cbar_kwargs={},
                                graticule=True,
                                graticule_kwargs={},
                                draw_contours=True,
                                contour_probs=[0.68],
                                contour_colors=['tab:orange'], # None -> pick colors from color scheme
                                zoom=True,
                                zoom_contained_prob_mass=0.99,
                                projection_type="zen_azi", # zen_azi or dec_ra
                                declination_trafo_function=None, # required to transform to dec/ra before plotting 
                                show_grid=False)

            direction_zoom.proj_plot([true_zenith[event].item()], [true_azimuth[event].item()],c='k',marker='x',label='true direction', zorder=2, s=150, linewidths=4)
            #direction_zoom.proj_plot([pred_mean_zenith[event].item()], [pred_mean_azimuth[event].item()],c='k',marker='x',label='predicted direction', zorder=2, s=150, linewidths=4)
            direction_zoom.legend(bbox_to_anchor=(0.2, 1.2), bbox_transform=direction_zoom.transAxes)
            """
            #table plotting
            col_labels = [r'Flavor [had. + EM %]',r'Energy$_{log}$ [$eV$]', r'Zenith [$deg$]',r'Azimuth [$deg$]', r"Area [$deg^2$]"]
            row_labels = ['predicted', 'true']
            table_vals =    [[np.round(100*flavor_pred[event].item(), 2), str(np.round(pred_energy[event], 2)) + r'$\pm$' +  str(np.round(np.sqrt(moments_dict_energy['varlike_0'].squeeze()[event]), 2)), np.round(np.rad2deg(pred_mean_zenith[event].item()), 2), np.round(np.rad2deg(pred_mean_azimuth[event].item()), 2), np.round(area_68, 2)],
                            [100* true_flavor[event].cpu().item(), np.round(true_energy[event].cpu().item(), 2), np.round(np.rad2deg(true_zenith[event].item()), 2) , np.round(np.rad2deg(true_azimuth[event].item()), 2), '-']
                            ]

            sum_table.axis("off")  
            my_table = sum_table.table(cellText=table_vals,
                                #colWidths=[0.2] * 3,
                                rowLabels=row_labels,
                                colLabels=col_labels,
                                #rowColours=row_colors,
                                loc='center right')
            """

            plt.savefig('/mnt/hdd/nheyer/gen2_shallow_reco/reconstruction/mix_split/results/' + args.Run_number + '/plots/' + args.version + '/event_MC.png', dpi=300)
            plt.close()
            quit()

            #break
        #break

