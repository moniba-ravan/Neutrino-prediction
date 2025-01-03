from astrotools import healpytools as hpt
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib
from radiotools import stats
from radiotools import plthelpers as php
from radiotools import helper 
import numpy as np
import matplotlib.patheffects as pe
from matplotlib import pylab
import argparse
import os
from NuRadioReco.utilities import units
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import metrics
import pandas as pd
from matplotlib.ticker import PercentFormatter
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--Run_number", type=str, default='RunXXX', help="Name a new Run to be saved")
parser.add_argument("--version", type=str, default='v1', help="Name a new Run to be saved")
args = parser.parse_args()


base_dir = '/mnt/hdd/moniba/final/Neutrino-prediction/results'

plots_path = base_dir + '/' + args.Run_number + '/plots/' + args.version + '/'
#plots_path = '/mnt/hdd/nheyer/gen2_shallow_reco/reconstruction/mix_split/results/' + args.Run_number + '/plots/test/'
results_path = base_dir + '/' + args.Run_number + '/full_y_pred/'

if(not os.path.exists(plots_path)):
    os.makedirs(plots_path)

result_dict = np.load(results_path + 'results-' + args.version + '.npy', allow_pickle=True).item()
#result_dict = np.load(results_path + 'results' + args.version + '_test.npy', allow_pickle=True).item()
#result_dict = np.load(results_path + 'results-' + args.version + '.npy', allow_pickle=True).item()
#result_dict = np.load(results_path + 'results-v2_train.npy', allow_pickle=True).item()

print(result_dict.keys())

true_zenith = result_dict['true_zenith']
true_azimuth = result_dict['true_azimuth']
pred_mean_zenith = result_dict['pred_mean_zenith']
pred_mean_azimuth = result_dict['pred_mean_azimuth']
pred_max_zenith = result_dict['pred_max_zenith']
pred_max_azimuth = result_dict['pred_max_azimuth']
true_energy = result_dict['true_energy']
pred_energy = result_dict['pred_energy']
pred_energy_std = np.sqrt(result_dict['pred_energy_std'])
true_flavor = result_dict['true_flavor']
pred_flavor = result_dict['pred_flavor']
pred_area_68 = result_dict['pred_area_68']
pred_area_50 = result_dict['pred_area_50']
#cov_direction_percentage = result_dict['cov_direction_percentage']#
#cov_energy_percentage = result_dict['cov_energy_percentage']#
#pred_direction_entopy = result_dict['pred_direction_entopy']
cov_energy_approx = result_dict['cov_energy_approx']
cov_direction_approx = result_dict['cov_direction_approx']
cov_direction_exact = result_dict['cov_direction_exact']

conversion = 3282.80635
percentile_68 = 0.68268
angle_difference_data_mean = np.array([helper.get_angle(helper.spherical_to_cartesian(pred_mean_zenith[i], pred_mean_azimuth[i]), helper.spherical_to_cartesian(true_zenith[i], true_azimuth[i])) for i in range(len(pred_mean_zenith))]) / units.deg
angle_difference_data_max = np.array([helper.get_angle(helper.spherical_to_cartesian(pred_max_zenith[i], pred_max_azimuth[i]), helper.spherical_to_cartesian(true_zenith[i], true_azimuth[i])) for i in range(len(pred_mean_zenith))]) / units.deg

def forward(x):
    return np.arccos(1 - ((x * np.pi * 0.5) / (180.**2))) * 360/(2*np.pi)

def inverse(x):
    return (1 - np.cos(x * (2*np.pi)/360)) * 2 * 180.**2 / np.pi 

def reorderLegend(ax=None,order=None,unique=False):
    if ax is None: ax=plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0])) # sort both labels and handles by labels
    if order is not None: # Sort according to a given list (not necessarily complete)
        keys=dict(zip(order,range(len(order))))
        labels, handles = zip(*sorted(zip(labels, handles), key=lambda t,keys=keys: keys.get(t[0],np.inf)))
    if unique:  labels, handles= zip(*unique_everseen(zip(labels,handles), key = labels)) # Keep only the first of each handle
    ax.legend(handles, labels)
    return(handles, labels)

def unique_everseen(seq, key=None):
    seen = set()
    seen_add = seen.add
    return [x for x,k in zip(seq,key) if not (k in seen or seen_add(k))]

def calc_coverage(exp, cov_percen):
    list_cov = []
    for idx, prob in enumerate(exp):
        list_cov.append(np.sum(cov_percen < prob) / len(cov_percen))
    return np.array(list_cov), exp

def get_histogram2d(x=None, y=None, z=None,
            bins=10, range=None,
            xscale="linear", yscale="linear", cscale="linear",
            normed=False, cmap=None, clim=(None, None),
            ax1=None, grid=True, shading='flat', colorbar={},
            cbi_kwargs={'orientation': 'vertical'},
            xlabel="", ylabel="", clabel="", title="",
            fname="hist2d.png"):
    """
    creates a 2d histogram
    Parameters
    ----------
    x, y, z :
        x and y coordinaten for z value, if z is None the 2d histogram of x and z is calculated
    numpy.histogram2d parameters:
        range : array_like, shape(2,2), optional
        bins : int or array_like or [int, int] or [array, array], optional
    ax1: mplt.axes
        if None (default) a olt.figure is created and histogram is stored
        if axis is give, the axis and a pcolormesh object is returned
    colorbar : dict
    plt.pcolormesh parameters:
        clim=(vmin, vmax) : scalar, optional, default: clim=(None, None)
        shading : {'flat', 'gouraud'}, optional
    normed: string
        colum, row, colum1, row1 (default: None)
    {x,y,c}scale: string
        'linear', 'log' (default: 'linear')
    """

    if z is None and (x is None or y is None):
        sys.exit("z and (x or y) are all None")

    if ax1 is None:
        fig, ax = plt.subplots(1)
        fig.subplots_adjust(wspace=0.3, left=0.05, right=0.95)
    else:
        ax = ax1

    if z is None:
        z, xedges, yedges = np.histogram2d(x, y, bins=bins, range=range)
        z = z.T
    else:
        xedges, yedges = x, y

    if normed:
        if normed == "colum":
            z = z / np.sum(z, axis=0)
        elif normed == "row":
            z = z / np.sum(z, axis=1)[:, None]
        elif normed == "colum1":
            z = z / np.amax(z, axis=0)
        elif normed == "row1":
            z = z / np.amax(z, axis=1)[:, None]
        else:
            sys.exit("Normalisation %s is not known.")

    color_norm = mpl.colors.LogNorm() if cscale == "log" else None
    vmin, vmax = clim
    im = ax.pcolormesh(xedges, yedges, z, shading=shading, vmin=vmin, vmax=vmax, norm=color_norm, cmap=cmap)

    if colorbar is not None:
        cbi = plt.colorbar(im, **cbi_kwargs)
        cbi.ax.tick_params(axis='both', **{"labelsize": 14})
        cbi.set_label(clabel)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    ax.set_title(title)

    return fig, ax, im

def coverage_plot(value = 'energy', coverage_list = cov_energy_approx):

    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, dpi=300)
    fig.set_size_inches(4, 4)
    fig.subplots_adjust(hspace=0)

    exponential = np.linspace(0,1,100)

    mask_nc = true_flavor == 0.0
    mask_cc = true_flavor == 1.0

    coverage_all, _ = calc_coverage(exponential, coverage_list)
    coverage_nc, _ = calc_coverage(exponential, coverage_list[mask_nc])
    coverage_cc, _ = calc_coverage(exponential, coverage_list[mask_cc])

    axs[0].grid()
    axs[0].plot([0, 100], [0, 100], 'r--', label='perfect coverage', linewidth=3)
    axs[0].plot(exponential*100, coverage_all*100, label='all events', linewidth=3)
    axs[0].plot(exponential*100, coverage_nc*100, label='nc events', linewidth=3)
    axs[0].plot(exponential*100, coverage_cc*100, label='cc events', linewidth=3)
    axs[0].set(ylabel='true coverage [%]')
    axs[0].legend()
    
    axs[1].grid()
    axs[1].set_axisbelow(True)
    axs[1].plot([0, 100], [0, 0], 'r--', linewidth=3)
    axs[1].plot(exponential*100, coverage_all*100-exponential*100, linewidth=3)
    axs[1].plot(exponential*100, coverage_nc*100-exponential*100, linewidth=3)
    axs[1].plot(exponential*100, coverage_cc*100-exponential*100, linewidth=3)
    axs[1].set(ylabel='true - expected [%]')
    plt.xlabel("expected coverage [%]")

    plt.tight_layout()
    
    plt.savefig(plots_path + value + '_coverage.png', dpi=300)
    plt.close()

def coverage_plot_per_energy(value = 'energy', interact = 'nc', coverage_list = cov_energy_approx, step_size=0.2, limits = [16, 20.2]):

    fig, axs = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]}, dpi=300)
    fig.set_size_inches(4, 4)
    fig.subplots_adjust(hspace=0)

    for i in np.arange(limits[0], limits[1], step_size):

        energy_mask = (true_energy >= i) & (true_energy <= i + step_size)

        #energy_list.append(i + step_size/2)

        cut_value_list = coverage_list[energy_mask]
        cut_flavor = true_flavor[energy_mask]

        mask_nc = cut_flavor == 0.0
        mask_cc = cut_flavor == 1.0

        exponential = np.linspace(0,1,100)

        coverage_all, _ = calc_coverage(exponential, cut_value_list)
        coverage_nc, _ = calc_coverage(exponential, cut_value_list[mask_nc])
        coverage_cc, _ = calc_coverage(exponential, cut_value_list[mask_cc])

        if interact == 'all':
            axs[0].plot(exponential*100, coverage_all*100, label='nc + cc ' + str(i) + ' - ' + str(i + step_size), linewidth=3)
            axs[1].plot(exponential*100, coverage_all*100-exponential*100, linewidth=3)
        if interact == 'nc':
            axs[0].plot(exponential*100, coverage_nc*100, label='nc ' + str(i) + ' - ' + str(i + step_size), linewidth=3)
            axs[1].plot(exponential*100, coverage_nc*100-exponential*100, linewidth=3)
        if interact == 'cc':
            axs[0].plot(exponential*100, coverage_cc*100, label='cc ' + str(i) + ' - ' + str(i + step_size), linewidth=3)
            axs[1].plot(exponential*100, coverage_cc*100-exponential*100, linewidth=3)


    axs[0].grid()
    axs[0].plot([0, 100], [0, 100], 'r--', label='perfect coverage', linewidth=3)

    axs[0].set(ylabel='true coverage [%]')
    axs[0].legend()
    
    axs[1].grid()
    axs[1].set_axisbelow(True)
    axs[1].plot([0, 100], [0, 0], 'r--', linewidth=3)
    
    axs[1].set(ylabel='true - expected [%]')
    plt.xlabel("expected coverage [%]")

    plt.tight_layout()
    
    plt.savefig(plots_path + value + '_coverage_per_energy_' + interact + '.png', dpi=300)
    plt.close()


def pred_vs_test(value = 'energy', truth = true_energy, predicted = pred_energy, bins = 30):

    #pred_vs_test 2d hist
    plt.figure(dpi=300)
    cmap = "BuPu"
    x_lab = ''
    y_lab = ''

    if value == 'energy':
        x_lab = "$\log_{10}\:(E_{true})$"
        y_lab = "$\log_{10}\:(E_{pred})$"
    if (value == 'zenith_max') or (value == 'zenith_mean'):
        x_lab = "$\Phi_{true} [deg]$"
        y_lab = "$\Phi_{pred} [deg]$"
    if (value == 'azimuth_max') or (value == 'azimuth_mean'):
        x_lab = "$\Theta_{true} [deg]$"
        y_lab = "$\Theta_{pred} [deg]$"


    fig, ax, im = get_histogram2d(truth, predicted, xlabel=x_lab, ylabel=y_lab, bins=bins, cmap=cmap)

    plt.plot([np.min(truth), np.max(truth)], [np.min(truth), np.max(truth)], c='k', linestyle = 'dashed')
    #plt.xlim(np.min(truth), np.max(truth))
    #plt.ylim(np.min(truth), np.max(truth))
    plt.tight_layout()
    plt.savefig(plots_path + value + '_pred_vs_test.png', dpi=300)
    plt.close()

################# ENERGY #################

def energy_diff():

    mask_165_195 = (true_energy >= 16.5) & (true_energy <= 19.5)
    mask_170_190 = (true_energy >= 17.0) & (true_energy <= 19.0)

    diff_all = pred_energy - true_energy
    diff_165_195 = pred_energy[mask_165_195] - true_energy[mask_165_195]
    diff_170_190 = pred_energy[mask_170_190] - true_energy[mask_170_190]

    fig1 = plt.figure(dpi=300)
    ax = plt.axes()

    php.get_histogram(diff_all, bins=np.linspace(-1.5, 1.5, 90)
    , xlabel="$\log_{10}\:(E_{pred}) - \log_{10}\:(E_{true})$"
    , ylabel='Events'
    , kwargs={'alpha': 0.7, 'facecolor': 'tab:blue', 'edgecolor': 'k'}
    , ax=ax
    , stats=False
    , overflow=False )
    php.plot_hist_stats(ax, diff_all, weights=None, posx=0.05, posy=0.95, overflow=None, underflow=None, rel=False, additional_text='', additional_text_pre='$16.0 < \log_{10}\:(E_{true})$ < 20.2', fontsize=10, color='tab:blue', va='top', ha='left', median=True, quantiles=True, mean=True, std=True, N=True)
    
    
    php.get_histogram(diff_165_195, bins=np.linspace(-1.5, 1.5, 90)
    , xlabel="$\log_{10}\:(E_{pred}) - \log_{10}\:(E_{true})$"
    , ylabel='Events'
    , kwargs={'alpha': 0.7, 'facecolor': 'tab:orange', 'edgecolor': 'k'}
    , ax=ax
    , stats=False
    , overflow=False )
    php.plot_hist_stats(ax, diff_165_195, weights=None, posx=0.95, posy=0.95, overflow=None, underflow=None, rel=False, additional_text='', additional_text_pre='$16.5 < \log_{10}\:(E_{true})$ < 19.5', fontsize=10, color='tab:orange', va='top', ha='right', median=True, quantiles=True, mean=True, std=True, N=True)
    
    php.get_histogram(diff_170_190, bins=np.linspace(-1.5, 1.5, 90)
    , xlabel="$\log_{10}\:(E_{pred}) - \log_{10}\:(E_{true})$"
    , ylabel='Events'
    , kwargs={'alpha': 0.7, 'facecolor': 'tab:green', 'edgecolor': 'k'}
    , ax=ax
    , stats=False
    , overflow=False )
    php.plot_hist_stats(ax, diff_170_190, weights=None, posx=0.05, posy=0.65, overflow=None, underflow=None, rel=False, additional_text='', additional_text_pre='$17.0 < \log_{10}\:(E_{true})$ < 19.0', fontsize=10, color='tab:green', va='top', ha='left', median=True, quantiles=True, mean=True, std=True, N=True)
    plt.ylim(0,6000)
    

    plt.savefig(plots_path + 'energy_diff.png', dpi=300)
    plt.close()

def energy_pred_std():

    mask_165_195 = (true_energy >= 16.5) & (true_energy <= 19.5)
    mask_170_190 = (true_energy >= 17.0) & (true_energy <= 19.0)

    std_165_195 = pred_energy_std[mask_165_195]
    std_170_190 = pred_energy_std[mask_170_190]

    fig1 = plt.figure(dpi=300)
    ax = plt.axes()

    php.get_histogram(pred_energy_std, bins=np.linspace(0, 1.5, 90)
    , xlabel="$\log_{10}(\sigma_{pred})$"
    , ylabel='Events'
    , kwargs={'alpha': 0.7, 'facecolor': 'tab:blue', 'edgecolor': 'k'}
    , ax=ax
    , stats=False
    , overflow=False )
    php.plot_hist_stats(ax, pred_energy_std, weights=None, posx=0.95, posy=0.95, overflow=None, underflow=None, rel=False, additional_text='', additional_text_pre='$16.0 < \log_{10}\:(E_{true})$ < 20.2', fontsize=10, color='tab:blue', va='top', ha='right', median=True, quantiles=True, mean=True, std=True, N=True)
    
    
    php.get_histogram(std_165_195, bins=np.linspace(0, 1.5, 90)
    , xlabel="$\log_{10}(\sigma_{pred})$"
    , ylabel='Events'
    , kwargs={'alpha': 0.7, 'facecolor': 'tab:orange', 'edgecolor': 'k'}
    , ax=ax
    , stats=False
    , overflow=False )
    php.plot_hist_stats(ax, std_165_195, weights=None, posx=0.95, posy=0.65, overflow=None, underflow=None, rel=False, additional_text='', additional_text_pre='$16.5 < \log_{10}\:(E_{true})$ < 19.5', fontsize=10, color='tab:orange', va='top', ha='right', median=True, quantiles=True, mean=True, std=True, N=True)
    
    php.get_histogram(std_170_190, bins=np.linspace(0, 1.5, 90)
    , xlabel="$\log_{10}(\sigma_{pred})$"
    , ylabel='Events'
    , kwargs={'alpha': 0.7, 'facecolor': 'tab:green', 'edgecolor': 'k'}
    , ax=ax
    , stats=False
    , overflow=False )
    php.plot_hist_stats(ax, std_170_190, weights=None, posx=0.95, posy=0.35, overflow=None, underflow=None, rel=False, additional_text='', additional_text_pre='$17.0 < \log_{10}\:(E_{true})$ < 19.0', fontsize=10, color='tab:green', va='top', ha='right', median=True, quantiles=True, mean=True, std=True, N=True)
    plt.ylim(0,4000)
    

    plt.savefig(plots_path + 'energy_std.png', dpi=300)
    plt.close()

def energy_pull():

    mask_165_195 = (true_energy >= 16.5) & (true_energy <= 19.5)
    mask_170_190 = (true_energy >= 17.0) & (true_energy <= 19.0)

    pull_all = (pred_energy - true_energy) / pred_energy_std
    pull_165_195 = (pred_energy[mask_165_195] - true_energy[mask_165_195]) / pred_energy_std[mask_165_195]
    pull_170_190 = (pred_energy[mask_170_190] - true_energy[mask_170_190]) / pred_energy_std[mask_170_190]

    fig1 = plt.figure(dpi=300)
    ax = plt.axes()

    php.get_histogram(pull_all, bins=np.linspace(-4, 4, 90)
    , xlabel="$\log_{10}\:(E_{pred}) - \log_{10}\:(E_{true}) \:/ \:\log_{10}\:(\sigma_{pred})$"
    , ylabel='Events'
    , kwargs={'alpha': 0.7, 'facecolor': 'tab:blue', 'edgecolor': 'k'}
    , ax=ax
    , stats=False
    , overflow=False )
    php.plot_hist_stats(ax, pull_all, weights=None, posx=0.05, posy=0.95, overflow=None, underflow=None, rel=False, additional_text='', additional_text_pre='$16.0 < \log_{10}\:(E_{true})$ < 20.2', fontsize=10, color='tab:blue', va='top', ha='left', median=True, quantiles=True, mean=True, std=True, N=True)
    
    
    php.get_histogram(pull_165_195, bins=np.linspace(-4, 4, 90)
    , xlabel="$\log_{10}\:(E_{pred}) - \log_{10}\:(E_{true}) \:/ \:\log_{10}\:(\sigma_{pred})$"
    , ylabel='Events'
    , kwargs={'alpha': 0.7, 'facecolor': 'tab:orange', 'edgecolor': 'k'}
    , ax=ax
    , stats=False
    , overflow=False )
    php.plot_hist_stats(ax, pull_165_195, weights=None, posx=0.95, posy=0.95, overflow=None, underflow=None, rel=False, additional_text='', additional_text_pre='$16.5 < \log_{10}\:(E_{true})$ < 19.5', fontsize=10, color='tab:orange', va='top', ha='right', median=True, quantiles=True, mean=True, std=True, N=True)
    
    php.get_histogram(pull_170_190, bins=np.linspace(-4, 4, 90)
    , xlabel="$\log_{10}\:(E_{pred}) - \log_{10}\:(E_{true}) \:/ \:\log_{10}\:(\sigma_{pred})$"
    , ylabel='Events'
    , kwargs={'alpha': 0.7, 'facecolor': 'tab:green', 'edgecolor': 'k'}
    , ax=ax
    , stats=False
    , overflow=False )
    php.plot_hist_stats(ax, pull_170_190, weights=None, posx=0.05, posy=0.65, overflow=None, underflow=None, rel=False, additional_text='', additional_text_pre='$17.0 < \log_{10}\:(E_{true})$ < 19.0', fontsize=10, color='tab:green', va='top', ha='left', median=True, quantiles=True, mean=True, std=True, N=True)
    plt.ylim(0,4000)
    

    plt.savefig(plots_path + 'energy_pull.png', dpi=300)
    plt.close()

################# FLAVOR #################

def flavor_histogram():

    fig = plt.figure(dpi=300)
    nc_predict = []
    cc_predict = []

    for i in range(true_flavor.shape[0]):
        if true_flavor[i] == 1:
            cc_predict.append(pred_flavor[i])
        else:
            nc_predict.append(pred_flavor[i])

    bins = np.linspace(0, 1, 40)

    counts_cc, _ = np.histogram(cc_predict, bins=40)
    plt.hist(cc_predict, label = 'cc = 1', alpha=0.5, bins=bins)

    counts_nc, _ = np.histogram(nc_predict, bins=40)
    plt.hist(nc_predict, label = 'nc = 0', alpha=0.5, bins=bins)
    plt.yscale('log')
    plt.xlabel('output')
    plt.ylabel('counts')
    plt.legend()
    plt.tight_layout()

    plt.savefig(plots_path + 'flavor_hist.png', dpi=300)
    plt.close()

def flavor_ROC():

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    step_size=0.4

    #for i in np.arange(16, 20.2, step_size):
    for i in np.arange(16.2, 20.2, step_size):

        energy_mask = (true_energy >= i) & (true_energy <= i + step_size)
        #energy_mask = energy_mask.squeeze()

        cut_pred = pred_flavor[energy_mask]
        cut_true = true_flavor[energy_mask]



        fpr, tpr, _ = metrics.roc_curve(cut_true,  cut_pred)

        if i == 20.0:
            plt.plot(fpr,tpr, label= '20.0 - 20.2 eV')
        else:
            plt.plot(fpr,tpr, label= str(round(i, 2)) + ' - ' + str(round(i + step_size, 2)) + ' eV')

    plt.plot([0, 1], [0, 1], 'k--', label = '50/50')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend()

    tick_list = np.linspace(0, 1, 21)

    plt.xticks(tick_list, rotation='vertical')
    plt.yticks(tick_list)

    plt.grid()

    plt.tight_layout()
    plt.savefig(plots_path + 'flavor_roc.png', dpi=300)
    plt.close()


def flavor_confusion_oscar(fp = 0.1):
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))


    #y_predicted = np.round(pred_flavor)
    cut_values = np.linspace(0, 1, 10000)[::-1]
    for cut in cut_values:
        y_predicted = np.where(pred_flavor < cut, 0, 1)

        energy_mask = (true_energy >= 18.0) & (true_energy <= 18.2)
        #energy_mask = energy_mask.squeeze()

        cut_pred = y_predicted[energy_mask]
        cut_true = true_flavor[energy_mask]

        cm = confusion_matrix(cut_true, cut_pred , normalize='true')
        if np.round(cm[0, 1], 4) == fp:
            print('cut', cut)
            print('cut', cm[0, 1])
            break
        #quit()
    labels = np.array([0, 1])
    classes = np.array(['had.', 'had. + EM'])

    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%' % (p)
            else:
                annot[i, j] = '%.1f%%' % (p)
    cm = confusion_matrix(cut_true, cut_pred , normalize='true')
    cm = confusion_matrix(cut_true, cut_pred, labels=labels, normalize='true')
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = pd.DataFrame(cm, index=labels, columns=labels)
    cm = cm * 100
    cm.index.name = 'True shower type'
    cm.columns.name = 'Predicted shower type'
    plt.yticks(va='center')

    ax = sns.heatmap(cm, annot=annot, fmt='', ax=axes, xticklabels=classes, square=True, cbar=True, cbar_kws={'format':PercentFormatter()}, yticklabels=classes, cmap="Blues")
    ax.collections[0].set_clim(0,100) 
    ax.set_xlabel(ax.get_xlabel(), fontdict={'weight': 'bold'})
    ax.set_ylabel(ax.get_ylabel(), fontdict={'weight': 'bold'})
    ax.set_title('18.0 - 18.2 log E')
    plt.savefig(plots_path + 'confusion_oscar_epoch.png', dpi=300)
    plt.close()




def flavor_confusion():
    fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(5,15))
    step_size=0.4

    y_predicted = np.round(pred_flavor)

    row = 0
    column = 0

    for i in np.arange(16.2, 20.2, step_size):

        energy_mask = (true_energy >= i) & (true_energy <= i + step_size)
        #energy_mask = energy_mask.squeeze()

        cut_pred = y_predicted[energy_mask]
        cut_true = true_flavor[energy_mask]

        cm = confusion_matrix(cut_true, cut_pred , normalize='true')

        axes[row, column].set_title(str(round(i, 2)) + ' - ' + str(round(i + step_size, 2)) + ' eV')
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)

        disp.plot(ax=axes[row, column], cmap = 'Blues')
        axes[row, column].set_xticklabels(['rest', r'$\nu_e CC$'])
        axes[row, column].set_yticklabels(['rest', r'$\nu_e CC$']) 
        disp.im_.set_clim(0, 1)
        disp.im_.set_clim(0, 1)


        if row==4:
            column +=1 
            row = 0
        else:
            row +=1
        
    
    plt.tight_layout()
    plt.savefig(plots_path + 'flavor_confusion.png', dpi=300)
    plt.close()

################# DIRECTION #################

def direction_area_hist(percentile = '68', bins = np.linspace(0, 1000, 100), over=1000000000000000, print_over = True, energy_lim = [17.5, 18.5]):

    if percentile == '50':
        area = pred_area_50 * conversion
        x_lab = "predicted 50$\%$ uncertainty area per event [deg$^2$]"
    elif percentile == '68':
        area = pred_area_68 * conversion
        x_lab = "predicted 68$\%$ uncertainty area per event [deg$^2$]"


    energy_mask = (true_energy >= energy_lim[0]) & (true_energy <= energy_lim[1])

    energy_cut = true_energy[energy_mask]
    area_cut = area[energy_mask]

    N = area_cut.size
    weights = np.ones(N)
    area_50 = stats.quantile_1d(area_cut, weights, 0.50)
    area_16 = stats.quantile_1d(area_cut, weights, 0.16)
    area_84 = stats.quantile_1d(area_cut, weights, 0.84)

    print(percentile)
    print(area_84)

    fig = plt.figure(dpi=300)
    fig.set_size_inches(5, 4)
    ax = plt.axes()
    
    php.get_histogram(area_cut, bins=bins
    #, xlabel="$Area_{68} [deg^2]$"
    , xlabel=x_lab
    , ylabel='Events'
    , kwargs={'alpha': 0.7, 'facecolor': 'tab:blue', 'edgecolor': 'k'}
    , ax=ax
    , stats=print_over
    , overflow=print_over
    )

    php.plot_hist_stats(ax, area_cut, weights=None, posx=0.35, posy=0.95, overflow=over, underflow=False, rel=False, additional_text='', additional_text_pre= str(energy_lim[0]) + '$ < \log_{10}\:(E_{sh, true})$ < ' + str(energy_lim[1]), fontsize=10, color='tab:blue', va='top', ha='left', median=False, quantiles=True, mean=False, std=False, N=True)
    php.plot_hist_stats(ax, area_cut, weights=None, posx=0.35, posy=0.74, overflow=False, underflow=False, rel=False, additional_text='', additional_text_pre='', fontsize=10, color='tab:red', va='top', ha='left', median=True, quantiles=True, mean=False, std=False, N=False)
    plt.axvline(area_50, color='tab:red', linestyle='dashed', linewidth=3)
    plt.axvline(area_84, color='tab:red', linestyle='dotted', linewidth=1.5)
    plt.axvline(area_16, color='tab:red', linestyle='dotted', linewidth=1.5)
    #plt.text(40., 1000., '16%', ha="left", va="center", c='tab:red', fontsize=10)
    #plt.text(200, 600., '50%', ha="left", va="center", c='tab:red', fontsize=10)
    #plt.text(590, 500., '84%', ha="left", va="center", c='tab:red', fontsize=10)

    #plt.text(745., 100., 'IceCube-Gen2 Preliminary', ha="right", va="bottom", c='tab:red', fontsize=10, rotation=90)
    secax = ax.secondary_xaxis('top', functions=(forward, inverse))
    secax.set_xlabel('Equivalent 1D angle [$^\circ$]')

    
    secax.set_xticks([0, 3, 5, 7, 9, 11, 13, 15])
    plt.xticks([0, 100, 300, 500, 700])
    #secax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])

    #plt.ylim(0,1200)
    
    plt.tight_layout()

    plt.savefig(plots_path + 'direction_area_' + percentile + '.png', dpi=300)
    plt.close()

def direction_sad_hist(type_angle = 'mean', bins = np.linspace(0, 1000, 100), over=1000000000000000, print_over = True, energy_lim = [17.5, 18.5]):

    if type_angle == 'mean':
        angle = angle_difference_data_mean
    elif type_angle == 'max':
        angle = angle_difference_data_max


    energy_mask = (true_energy >= energy_lim[0]) & (true_energy <= energy_lim[1])

    energy_cut = true_energy[energy_mask]
    angle_cut = angle[energy_mask]

    N = angle_cut.size
    weights = np.ones(N)
    angle_50 = stats.quantile_1d(angle_cut, weights, 0.50)
    angle_16 = stats.quantile_1d(angle_cut, weights, 0.16)
    angle_84 = stats.quantile_1d(angle_cut, weights, 0.84)

    angle_68 = stats.quantile_1d(angle_cut, weights, 0.68)


    fig = plt.figure(dpi=300)
    fig.set_size_inches(5, 4)
    ax = plt.axes()
    
    php.get_histogram(angle_cut, bins=bins
    #, xlabel="$Area_{68} [deg^2]$"
    , xlabel="$\Psi[^\circ]$"
    , ylabel='Events'
    , kwargs={'alpha': 0.7, 'facecolor': 'tab:blue', 'edgecolor': 'k'}
    , ax=ax
    , stats=print_over
    , overflow=print_over
    )

    php.plot_hist_stats(ax, angle_cut, weights=None, posx=0.35, posy=0.95, overflow=over, underflow=False, rel=False, additional_text='', additional_text_pre= str(energy_lim[0]) + '$ < \log_{10}\:(E_{sh, true})$ < ' + str(energy_lim[1]), fontsize=10, color='tab:blue', va='top', ha='left', median=False, quantiles=True, mean=False, std=False, N=True)
    php.plot_hist_stats(ax, angle_cut, weights=None, posx=0.35, posy=0.74, overflow=False, underflow=False, rel=False, additional_text='', additional_text_pre='', fontsize=10, color='tab:red', va='top', ha='left', median=True, quantiles=True, mean=False, std=False, N=False)

    plt.axvline(angle_50, color='tab:red', linestyle='dashed', linewidth=3)
    plt.axvline(angle_84, color='tab:red', linestyle='dotted', linewidth=1.5)
    plt.axvline(angle_16, color='tab:red', linestyle='dotted', linewidth=1.5)
    plt.axvline(angle_68, color='k', linestyle='dotted', linewidth=1.5, label =f'68%')
    #plt.text(40., 1000., '16%', ha="left", va="center", c='tab:red', fontsize=10)
    #plt.text(200, 600., '50%', ha="left", va="center", c='tab:red', fontsize=10)
    #plt.text(590, 500., '84%', ha="left", va="center", c='tab:red', fontsize=10)

    #plt.text(745., 100., 'IceCube-Gen2 Preliminary', ha="right", va="bottom", c='tab:red', fontsize=10, rotation=90)
    #secax = ax.secondary_xaxis('top', functions=(forward, inverse))
    #secax.set_xlabel('Equivalent 1D angle [$^\circ$]')

    
    #secax.set_xticks([0, 3, 5, 7, 9, 11, 13, 15])
    #plt.xticks([0, 100, 300, 500, 700])
    #secax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14])

    #plt.ylim(0,1200)
    plt.legend(loc=7)
    
    plt.tight_layout()

    plt.savefig(plots_path + 'direction_sad_' + type_angle + '.png', dpi=300)
    plt.close()

################# RESOLUTION #################

def value_vs_energy(value= 'energy_bias', step_size=0.2, limits = [16, 20.2]):

    fig = plt.figure(dpi=300)
    fig.set_size_inches(5, 4)
    #fig.set_size_inches(4, 3)
    ax = plt.axes()

    if value == 'energy_div':
        pred_value = 10**pred_energy
        true_value = 10**true_energy
        comp_value = pred_value / true_value
        y_label = "$E_{sh,\:pred} \: / \: E_{sh,\:true}$"
        ax.set_ylim(bottom=0,top=3.5)

    if value == 'energy_diff':
        pred_value = pred_energy
        true_value = true_energy
        comp_value = pred_value - true_value
        y_label = "$\Delta (\log_{10}\:E_{sh})$"
        #ax.set_ylim(bottom=-0.2, top=0.2)
        plt.text(16.02, -0.45, 'biased energy', size = 9)
        plt.text(16.02, -0.5, 'region', size = 9)

    if value == 'energy_std':
        comp_value = pred_energy_std
        y_label = "$\hat{\sigma} \:(\log_{10}\:E_{sh})$"
        plt.text(16.02, 0.03, 'biased energy', size = 9)
        plt.text(16.02, 0.01, 'region', size = 9)
        ax.set_ylim(bottom=0,top=0.55)

    if value == 'direction_sad_mean':
        comp_value = angle_difference_data_mean
        y_label = '$\Psi_{mean}$ [$^\circ$]'

    if value == 'direction_sad_max':
        comp_value = angle_difference_data_max
        y_label = '$\Psi_{max}$ [$^\circ$]'

    if value == 'direction_area68':
        fig.set_size_inches(5.5, 4)
        comp_value = pred_area_68 * conversion 
        #plt.text(16.02, 1.5, 'biased energy', size = 9)
        #plt.text(16.02, 1.1, 'region', size = 9)
        y_label = "predicted 68$\%$ uncertainty area per event [deg$^2$]"


    if value == 'direction_area50':
        comp_value = pred_area_50 * conversion 
        y_label = "predicted 50$\%$ uncertainty area per event [deg$^2$]"

    energy_list = []

    nc_16_list = []
    nc_50_list = []
    nc_84_list = []

    cc_16_list = []
    cc_50_list = []
    cc_84_list = []


    for i in np.arange(limits[0], limits[1], step_size):
        #print(np.round(i))

        #if np.round(i, 2) not in [17.0, 18.0, 19.0]:
        #    continue

        energy_mask = (true_energy >= i) & (true_energy <= i + step_size)

        energy_list.append(i + step_size/2)

        cut_value = comp_value[energy_mask]
        cut_flavor = true_flavor[energy_mask]

        mask_nc = cut_flavor == 0.0
        mask_cc = cut_flavor == 1.0


        nc_16 = stats.quantile_1d(cut_value[mask_nc], np.ones(cut_value[mask_nc].size), 0.16)
        nc_50 = stats.quantile_1d(cut_value[mask_nc], np.ones(cut_value[mask_nc].size), 0.5)
        nc_84 = stats.quantile_1d(cut_value[mask_nc], np.ones(cut_value[mask_nc].size), 0.84)

        nc_16_list.append(nc_16)
        nc_50_list.append(nc_50)
        nc_84_list.append(nc_84)

        cc_16 = stats.quantile_1d(cut_value[mask_cc], np.ones(cut_value[mask_cc].size), 0.16)
        cc_50 = stats.quantile_1d(cut_value[mask_cc], np.ones(cut_value[mask_cc].size), 0.5)
        cc_84 = stats.quantile_1d(cut_value[mask_cc], np.ones(cut_value[mask_cc].size), 0.84)

        cc_16_list.append(cc_16)
        cc_50_list.append(cc_50)
        cc_84_list.append(cc_84)#plt.plot(x, y, marker=r'$\uparrow$')
    #plt.plot(x, y, marker=r'$\downarrow$')

    #plt.scatter([17, 18, 19], nc_50_list, color = 'tab:green', marker='x', label ='with birefringence', zorder=12.0)

    plt.errorbar(energy_list, nc_50_list, xerr=step_size/2, color = 'tab:blue', fmt='D', label ='had.', zorder=12.0)
    plt.errorbar(energy_list, cc_50_list, xerr=step_size/2, color = 'tab:red', fmt='D', label ='had. + EM', zorder=11.0)

    plt.scatter(energy_list, nc_16_list, color = 'tab:blue', marker=r'$\uparrow$', label ='16 percentile', zorder=15.0)
    plt.scatter(energy_list, cc_16_list, color = 'tab:red', marker=r'$\uparrow$', zorder=10.0)

    plt.scatter(energy_list, nc_84_list, color = 'tab:blue', marker=r'$\downarrow$', label ='84 percentile', zorder=20.0)
    plt.scatter(energy_list, cc_84_list, color = 'tab:red', marker=r'$\downarrow$', zorder=10.0)


    if value == 'direction_area68':
        print(ax.get_ylim())
        ax.set_ylim(bottom=1,top=1500)
        ax.set_yscale('log')
        secax = ax.secondary_yaxis('right', functions=(forward, inverse))
        secax.set_ylabel('Equivalent 1D angle [$^\circ$]')
        plt.yticks([1, 10, 100])
        secax.set_yticks([1, 10])
        
    #plt.axvspan(16, 17, alpha=0.2, color='k')
    plt.xlim(16, 20.2)
    plt.ylabel(y_label)
    plt.xlabel("$\log_{10}\:E_{sh, \: true}$")
    plt.grid(zorder=-1.0)
    #print('hello')

    plt.legend(loc=1)

    #reorderLegend(ax,['84 percentile', 'had. ', '16 percentile', 'had. + EM'])

    plt.tight_layout()
    plt.savefig(plots_path + value + '_vs_energy.png', dpi=300)
    #plt.savefig(plots_path + value + '_vs_energy_syst.png', dpi=300)
    plt.close()

def value_vs_energy68(value= 'energy_bias', step_size=0.2, limits = [16, 20.2]):

    fig = plt.figure(dpi=300)
    fig.set_size_inches(5, 4)
    fig.set_size_inches(4, 3)
    ax = plt.axes()

    if value == 'energy_bias':
        pred_value = pred_energy
        true_value = true_energy
        comp_value = pred_value - true_value
        y_label = "$median \: \Delta (\log_{10}\:E_{sh})$"

    if value == 'energy_std':
        comp_value = pred_energy_std
        y_label = "$\hat{\sigma} \:(\log_{10}\:E_{sh})$"

    if value == 'direction_sad_mean':
        comp_value = angle_difference_data_mean
        y_label = '$\Psi_{68\%}$ [$^\circ$]'

    if value == 'direction_sad_max':
        comp_value = angle_difference_data_max
        y_label = '$\Psi_{max}$ [$^\circ$]'

    if value == 'direction_area68':
        comp_value = pred_area_68 * conversion 
        y_label = "predicted 68$\%$ uncertainty area per event [deg$^2$]"

    if value == 'direction_area50':
        comp_value = pred_area_50 * conversion 
        y_label = "predicted 50$\%$ uncertainty area per event [deg$^2$]"

    energy_list = []

    nc_68_list = []
    cc_68_list = []

    for i in np.arange(limits[0], limits[1], step_size):

        if np.round(i, 2) not in [17.0, 18.0, 19.0]:
            continue

        energy_mask = (true_energy >= i) & (true_energy <= i + step_size)

        energy_list.append(i + step_size/2)

        cut_value = comp_value[energy_mask]
        cut_flavor = true_flavor[energy_mask]

        mask_nc = cut_flavor == 0.0
        mask_cc = cut_flavor == 1.0


        nc_68 = stats.quantile_1d(cut_value[mask_nc], np.ones(cut_value[mask_nc].size), percentile_68)

        nc_68_list.append(nc_68)

        cc_68 = stats.quantile_1d(cut_value[mask_cc], np.ones(cut_value[mask_cc].size), percentile_68)

        cc_68_list.append(cc_68)


    #plt.errorbar(energy_list, nc_68_list, xerr=step_size/2, color = 'tab:blue', fmt='D', label ='had.', zorder=10.0)
    #plt.errorbar(energy_list, cc_68_list, xerr=step_size/2, color = 'tab:red', fmt='D', label ='had. + EM', zorder=10.0)

    plt.scatter([17, 18, 19], [5.419613425394631, 2.9703108436144583, 1.9491259835630452], color = 'tab:red', marker='D', label ='baseline', zorder=10.0)
    plt.scatter([17, 18, 19], nc_68_list, color = 'tab:green', marker='x', label ='baseline', zorder=10.0)

    plt.ylabel(y_label)
    plt.xlabel("$\log_{10}\:E_{sh, \: true}$")
    plt.grid(zorder=-1.0)
    plt.ylim(0, 12)

    plt.legend()

    plt.tight_layout()
    plt.savefig(plots_path + value + '_vs_energy_68_syst.png', dpi=300)
    #plt.savefig(plots_path + value + '_vs_energy_68.png', dpi=300)
    plt.close()




def ROC_Christian():

    fig, axes = plt.subplots(1, 1, figsize=(5, 5))
    step_size=0.4


    colors = ['k', 'r', 'b', 'g', 'y']


    #for i in np.arange(16, 20.2, step_size):
    for indx, i in enumerate(np.arange(17.0, 19.0, step_size)):

        energy_mask = (true_energy >= i) & (true_energy <= i + step_size)
        #energy_mask = energy_mask.squeeze()

        cut_pred = np.array(pred_flavor)[energy_mask]
        cut_true = true_flavor[energy_mask]



        fpr, tpr, _ = metrics.roc_curve(cut_true,  cut_pred)

        if i == 20.0:
            #plt.plot(fpr,tpr, label= '20.0 - 20.2 eV')
            plt.plot(tpr, fpr, label= '20.0 - 20.2 eV')
        else:
            plt.plot(tpr, fpr, label= str(round(i, 2)) + ' - ' + str(round(i + step_size, 2)) + ' eV', color=colors[indx])

    plt.plot([0, 1], [0, 1], 'k--', label = '50/50')
    plt.xlabel(r'Fraction correctly identified $\nu_e$ CC (true positive)')
    plt.ylabel(r'Fraction incorrectly identified $\nu_e$ CC (false positive)')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()

    tick_list = np.linspace(0, 1, 21)

    plt.xticks(tick_list, rotation='vertical')
    plt.yticks(tick_list)

    plt.grid()

    plt.tight_layout()
    plt.savefig(plots_path + 'roc_Christian.png', dpi=300)
    plt.close()



coverage_plot_per_energy(value = 'energy', interact = 'cc', coverage_list = cov_energy_approx, step_size=0.5, limits = [16.0, 20.0])
coverage_plot_per_energy(value = 'energy', interact = 'nc', coverage_list = cov_energy_approx, step_size=0.5, limits = [16.0, 20.0])
coverage_plot_per_energy(value = 'direction', interact = 'cc', coverage_list = cov_direction_approx, step_size=0.5, limits = [16.0, 20.0])
coverage_plot_per_energy(value = 'direction', interact = 'nc', coverage_list = cov_direction_approx, step_size=0.5, limits = [16.0, 20.0])
coverage_plot_per_energy(value = 'direction_exact', interact = 'cc', coverage_list = cov_direction_exact, step_size=0.5, limits = [16.0, 20.0])
coverage_plot_per_energy(value = 'direction_exact', interact = 'nc', coverage_list = cov_direction_exact, step_size=0.5, limits = [16.0, 20.0])


coverage_plot('energy', cov_energy_approx )
coverage_plot('direction', cov_direction_approx )
coverage_plot('direction_exact', cov_direction_exact )
#coverage_plot('direction', cov_direction_percentage)
pred_vs_test('energy', true_energy, pred_energy)
pred_vs_test('zenith_max', np.rad2deg(true_zenith), np.rad2deg(pred_max_zenith))
pred_vs_test('zenith_mean', np.rad2deg(true_zenith), np.rad2deg(pred_mean_zenith))
pred_vs_test('azimuth_max', np.rad2deg(true_azimuth), np.rad2deg(pred_max_azimuth))
pred_vs_test('azimuth_mean', np.rad2deg(true_azimuth), np.rad2deg(pred_mean_azimuth))

energy_diff()
energy_pred_std()
energy_pull()

flavor_histogram()
flavor_ROC()
flavor_confusion()

direction_area_hist(percentile = '68', bins = np.linspace(0, 1000, 100), over=3196, print_over = False, energy_lim = [17.5, 18.5])
direction_area_hist(percentile = '50', bins = np.linspace(0, 750, 75), over=2300, print_over = False, energy_lim = [17.5, 18.5])
direction_area_hist(percentile = '68', bins = np.linspace(0, 1000, 100), over=3196, print_over = False, energy_lim = [16.0, 20.2])
direction_area_hist(percentile = '50', bins = np.linspace(0, 750, 75), over=2300, print_over = False, energy_lim = [16.0, 20.2])


direction_sad_hist(type_angle = 'mean', bins = np.linspace(0, 50, 100), over=2569, print_over = False, energy_lim = [16.0, 20.2])
direction_sad_hist(type_angle = 'max', bins = np.linspace(0, 50, 100), over=3756, print_over = False, energy_lim = [16.0, 20.2])


value_vs_energy(value= 'energy_div')

value_vs_energy(value= 'energy_std')

value_vs_energy(value= 'energy_diff')


value_vs_energy(value= 'direction_sad_mean')

value_vs_energy68(value= 'direction_sad_mean')

value_vs_energy(value= 'direction_sad_max')

value_vs_energy(value= 'direction_area68')

value_vs_energy(value= 'direction_area50')

ROC_Christian()



flavor_confusion_oscar()