#!/usr/bin/env python3
# encoding: utf-8
"""
This script shows how to perform multiview TFM.

Warning: this script can take up to several minutes to run and opens more than 20 windows.


"""
import logging
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import arim.settings as s
import arim
import arim.plot as aplt


arim.has_cuda_gpu.test()

#%% Output and parameters
interpolate_position='linear'
PLOT_TIME_TO_SURFACE = True
SHOW_RAY = False
PLOT_TFM = True
PLOT_INTERFACE = False
SAVEFIG = True
pixel_size=0.25e-3 # in metres

use_cpu=1 #Options are 1 (OpenMP+C CPU), 0 (CUDA+C GPU) None = Python + JIT

#%% Figure and logger
mpl.rcParams['image.cmap'] = 'viridis'
mpl.rcParams['figure.figsize'] = [12., 7.]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.info("Start script")


v_couplant = 1480.

#Aluminium
foldername = r'C:\RBData\arim-datasets\20160725\aluminium_immersion'
filename=r'exp_data_systemA_aluminium_notch_sample0_128element_dp2_avg_0032.mat'

if "/" not in foldername[-1]:
    foldername = foldername +'/'

expdata_filename = foldername + filename

v_longi = 6400
v_shear = v_longi*0.5
zdepth=40.18e-3
tmin=10e-6
tmax=180e-6
freqmin=3.0e6
freqmax=5.5e6
frame = arim.io.load_expdata(expdata_filename)
frame.probe  = arim.probes['ima_50_MHz_128_1d']

zmax=zdepth+5e-3

#Set probe reference point to first element
# put the first element in O(0,0,0), then it will be in (0,0,z) later.
frame.probe.locations.translate(-frame.probe.locations[0], inplace=True)

# -------------------------------------------------------------------------
#%% Registration: get the position of the probe from the pulse-echo data

# Prepare registration
filt = arim.signal.Hilbert() + \
       arim.signal.ButterworthBandpass(order=5, cutoff_min=freqmin, cutoff_max=freqmax, time=frame.time)
frame.apply_filter(filt)

ax, imag = aplt.plot_bscan_pulse_echo(frame,ax=None,clim=[-40,0],use_dB=True)
ax.figure.savefig(str(foldername)+"fig_BscanPulseEcho.png", bbox_inches='tight') 

# Detect frontwall:
time_to_surface = arim.registration.detect_surface_from_extrema(frame, tmin=tmin, tmax=tmax)

if PLOT_TIME_TO_SURFACE:
    plt.figure()
    plt.plot(time_to_surface[frame.tx==frame.rx])
    plt.xlabel('element')
    plt.ylabel('time (µs)')
    plt.gca().yaxis.set_major_formatter(aplt.us_formatter)
    plt.gca().yaxis.set_minor_formatter(aplt.us_formatter)

    plt.title('time between elements and frontwall - must be a line!')


# Move probe:
distance_to_surface = time_to_surface * v_couplant / 2
frame, iso = arim.registration.move_probe_over_flat_surface(frame, distance_to_surface, full_output=True)

logger.info('probe orientation: {:.2f}°'.format(np.rad2deg(iso.theta)))
logger.info('probe distance (min): {:.2f} mm'.format(-1e3*iso.z_o))

# -------------------------------------------------------------------------
#%% Define interfaces

numinterface = 1000
numinterface2 =1000

probe = frame.probe.locations
probe.name = 'Probe'

xmin = -10e-3
xmax = 80e-3

frontwall = arim.geometry.Points(
    x=np.linspace(xmin, xmax, numinterface),
    y=np.zeros((numinterface, ), dtype=np.float),
    z=np.zeros((numinterface, ), dtype=np.float),
    name='Frontwall')

backwall = arim.geometry.Points(
    x=np.linspace(xmin, xmax, numinterface2),
    y=np.zeros((numinterface2, ), dtype=np.float),
    z=np.full((numinterface2, ), zdepth, dtype=np.float),
    name='Backwall')


grid = arim.geometry.Grid(xmin, xmax,
                          ymin=0., ymax=0.,
                          zmin=0., zmax=zmax,
                          pixel_size=pixel_size)

print("Interfaces:")
for p in [probe, frontwall, backwall, grid.as_points]:
    print("\t{} \t\t{} points".format(p, len(p)))


#%% Plot interface
def plot_interface(title=None, show_grid=True, ax=None, element_normal=True):
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    interfaces = [probe, frontwall, backwall]
    if show_grid:
        interfaces += [grid.as_points]
    for (interface, marker) in zip(interfaces, ['o', '.', '.', '.k']):
        ax.plot(interface.x, interface.z, marker, label=interface.name)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_title(title)
    ax.grid()

    if element_normal:
        k = 4# one every k arrows
        ax.quiver(frame.probe.locations.x[::k],
                  frame.probe.locations.z[::k],
                  frame.probe.orientations.x[::k],
                  frame.probe.orientations.z[::k],
                  units='xy', angles='xy',
                  width=0.0003, color='c')


    ylim = ax.get_ylim()
    if ylim[0] < ylim[1]:
        ax.invert_yaxis()

    ax.axis('equal')
    fig.show()
    return ax

plot_interface("Interfaces", show_grid=False, element_normal=True)

# -------------------------------------------------------------------------
#%% Setup views
tfms1=arim.im.BaseMultiTFMviews(frame,probe,frontwall,backwall,grid,v_longi,v_shear,v_couplant,interpolate_position='nearest')

#%% Run all TFM

tic = time.clock() 
tfms1.BasicRunAll(use_cpu=use_cpu)
toc = time.clock()     
logger.info("Performed {} delay-and-sum's in {:.2f} s".format(len(tfms1.tfms), toc-tic))



#%% Plot all TFM
if PLOT_TFM:
    start_index = 0
    end_index = grid.as_points.closest_point(54e-3, 0.0, 21e-3)

    func_res = lambda x: arim.utils.decibel(x)
    #func_res = lambda x: np.abs(x)
    clim = [-40, 0]

    for i, tfm in enumerate(tfms1.tfms):
        view = tfm.view

        ax, _ = aplt.plot_tfm_generic(tfm.grid,tfm.MaskedResNaN, clim=clim,func_res=func_res,title=view.name)
        
        if PLOT_INTERFACE:
            ax = plot_interface(view.name, show_grid=False, ax=ax)
            ax.legend().remove()

        if SHOW_RAY:
            ray_tx = tfm.rays_tx.get_coordinates_one(start_index, end_index)
            ray_rx = tfm.rays_rx.get_coordinates_one(start_index, end_index)

            ax.plot(ray_tx.x, ray_tx.z, 'm--', label='TX')
            ax.plot(ray_rx.x, ray_rx.z, 'c-.', label='RX')
            ax.legend()
            ax.legend().remove()
        if SAVEFIG:
            ax.figure.savefig(str(foldername)+"fig_{:02}_{}.png".format(i, view.name), bbox_inches='tight')

    if SAVEFIG:
        plt.close("all")
        
    
   
# Block script until windows are closed.
#plt.show()