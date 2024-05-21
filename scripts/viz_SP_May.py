import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from astropy.table import Table #, QTable, hstack, vstack
import astropy.units as u
import astropy.coordinates as coord
from astropy.io import fits
from astropy.constants import G

import gala.coordinates as gc
import gala.potential as gp
import gala.dynamics as gd
from gala.dynamics import mockstream as ms
from gala.units import galactic

from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize

from functools import partial
import pickle
import time
import glob

import healpy as hp
from skimage import io, color

ham_mw = gp.Hamiltonian(gp.MilkyWayPotential())

#Added by SP                                                                                                
mpl.rcParams.update({'font.size': 26})
label_size = 16
mpl.rcParams['xtick.labelsize'] = 20#label_size                                                             
mpl.rcParams['ytick.labelsize'] =20 #label_size  

#added by SP
galstreams = np.loadtxt("../data/galstreams_rgc.txt")

# TNG best-fit potential
bulgePot = gp.HernquistPotential(10**10.26*u.Msun, 1.05*u.kpc, units=galactic)
# issues integrating orbits in a disk potential with vanishing b
diskPot = gp.MiyamotoNagaiPotential(10**10.70*u.Msun, 4.20*u.kpc, 0.006*u.kpc, units=galactic)
diskPot = gp.MiyamotoNagaiPotential(10**10.70*u.Msun, 4.20*u.kpc, 0.28*u.kpc, units=galactic)
haloPot = gp.NFWPotential(10**11.57*u.Msun, 11.59*u.kpc, c=0.943, units=galactic)

totalPot = gp.CCompositePotential(component1=bulgePot, component2=diskPot, component3=haloPot)
ham = gp.Hamiltonian(totalPot)

coord.galactocentric_frame_defaults.set('v4.0')
gc_frame = coord.Galactocentric()


def get_done(hid=523889, lowmass=True, target='progenitors', verbose=False, fstar=1):
    """Return indices of fully simulated streams"""
    #t = Table.read('../data/stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.{:d}.fits'.format(target, hid, lowmass,fstar))
    t = Table.read('../data/output_stream_progenitors_halo.{:d}_lowmass.1_fstar.-1.00.fits'.format(hid))
    #print(t)
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    
    to_run = ind_all
    ind_done = np.zeros(N, dtype=bool)
    
    if target=='progenitors':
        label = 'stream'
    else:
        label = 'gc'
    
    #fout = glob.glob('../data/streams/halo.{:d}_{:s}.{:.2f}*'.format(hid, label, fstar))

    #edited by SP to get current streams. 
    fout = glob.glob('../data/streams/halo.{:d}_{:s}.{:.2f}*'.format(hid, label, fstar)) 

    #print(fout)
    #    print(fstar)

    for f in fout:
#        print(f)
        #print(f)
        i = int(f.split('.')[-2])
        ind_done[i] = 1
    
    if verbose: print(np.sum(ind_done), N)
    
    return ind_done

  #SP tweaking for talk  
def plot_distant(hid=523889, lowmass=True, target='progenitors', test=False, rgal=15):
    """Diagnostic plot showing simulated stream particles (downsampled for more massive streams)"""
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.-1.00.fits'.format(target, hid, lowmass))
    #print(t.info())
    N = len(t)
    
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target)
    
    ind_dist = t['rgal_stream']>rgal
    #print(t.info)
    x_ = np.sqrt(t['med_l']**2 + t['med_b']**2)
    y_ = np.abs(t['lz']/t['l'])
    ind_stream = (x_>2) & (y_<0.95)
    
    to_plot = ind_all[ind_done & ind_dist & ind_stream]
    Nplot = np.size(to_plot)
    print(Nplot)
    if test:
        Nplot = 20


    ##### For paper values #####    
#    print('rperi')   
 #   print(np.mean(t['rperi'][to_plot]))
 #   print(np.min(t['rperi'][to_plot]))
 #   print(np.max(t['rperi'][to_plot]))
 #   print('mbirth')
 #   print(np.mean(t['logMgc_at_birth'][to_plot]))
 #   print(np.min(t['logMgc_at_birth'][to_plot]))
 #   print(np.max(t['logMgc_at_birth'][to_plot]))

    
    plt.close()
    fig = plt.figure(figsize=(14,14))
    ax1 = fig.add_subplot(211, projection='mollweide')
    ax2 = fig.add_subplot(212, projection='mollweide')
    
    plt.sca(ax2)
    plt.title('MW galstreams (20)')
    plt.xlabel('l [deg]')
    plt.ylabel('b [deg]')
    plt.text(0.85, 0.9, r'$d>'+str(rgal)+'$ kpc', transform=plt.gca().transAxes)
    
    plt.sca(ax1)
#    plt.xticks([])
    plt.xlabel('l [deg]')
    plt.ylabel('b [deg]')
    plt.title('Model streams ('+ str(Nplot)+')')#, g < ' +str(27) )
    plt.text(0.85, 0.98, r'$g<27.5$', transform=plt.gca().transAxes)
    plt.text(0.85, 0.9, r'$d>'+str(rgal)+'$ kpc', transform=plt.gca().transAxes)


    # to plot gal streams for 22 streams at rgal > 15 kpc
    for i in range(20):
        l_galstreams = np.loadtxt('../data/galstreams_rgal15/MW15tracks_l_{:d}.txt'.format(i))
        b_galstreams = np.loadtxt('../data/galstreams_rgal15/MW15tracks_b_{:d}.txt'.format(i))

        ax2.plot(l_galstreams,b_galstreams, 'o', color='k', mew=0, ms=1, alpha=0.1, rasterized=True)

#    j = 0
    #to plot model streams
    for i in range(Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.1.00.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))   

        cg = pkl['cg']
        
        ind_sdss = pkl['g']<22.5
        ind_lsst = pkl['g']<27.5
        
        color = 'k'
        

        #checks if there are any starts above the gband lsst limit.
        #and then only plots those stars by using ind_lsst
        if np.sum(ind_lsst):
            
            plt.sca(ax1)
            plt.plot(cg.l.wrap_at(180*u.deg).radian[ind_lsst], cg.b.radian[ind_lsst], 'o', color=color, mew=0, ms=1, alpha=0.1, rasterized=True)

    plt.tight_layout()
    plt.savefig('../plots/lsst_galstreams_rgal.{:.0f}_SP.pdf'.format(rgal),dpi=300)


    plt.close()
    fig = plt.figure(figsize=(7,7))
#    #sp adding to get main sequence turnoff 
    for i in range(Nplot):
        # pkl = pickle.load(open('../data/streams_notfull_butgood/halo.{:d}_stream.-1.00.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))        
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.1.00.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))        
        cg = pkl['cg']
        g =  pkl['g']
        r = pkl['r']

        plt.scatter(g-r,g, label='CMD for stream ' + str(to_plot[i]), marker='.', color='black',alpha=0.3)
        plt.xlabel('$g-r$', fontsize='large')
        plt.ylabel('$g$(mag)', fontsize='large')
        plt.axhline(27.5, linestyle='--', color ='black',label='LSST limiting mag')
        plt.xlim([0,3])
        plt.legend(fontsize='small', loc='upper right')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        print(i)
        plt.xlim([0,1])
        plt.savefig('../plots/lsst_CMDs/lsst_cmd_.{:.0f}_stream{:04d}.pdf'.format(rgal,to_plot[i]),dpi=300)
        plt.close()
        

def plot_cmds(hid=523889, lowmass=True, target='progenitors', test=False, rgal=15):
    """"""
    
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.-1.00.fits'.format(target, hid, lowmass))
    #print(t.info())
    N = len(t)
    
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target)
    
    ind_dist = t['rgal_stream']>rgal
    #print(t.info)
    x_ = np.sqrt(t['med_l']**2 + t['med_b']**2)
    y_ = np.abs(t['lz']/t['l'])
    ind_stream = (x_>2) & (y_<0.95)
    
    to_plot = ind_all[ind_done & ind_dist & ind_stream]
    Nplot = np.size(to_plot)
    print(Nplot)
    
    if test:
        Nplot = 2
    
    pp = PdfPages('../plots/cmds_halo.{:d}_rgal15.pdf'.format(hid))

    for i in range(Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.1.00.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        g = pkl['g']
        r = pkl['r']

        plt.close()
        fig = plt.figure(figsize=(5,10))
        
        plt.plot(g-r, g, 'ko', ms=2, mew=0, alpha=0.1, label='Stream {:d}'.format(to_plot[i]), rasterized=True)
        plt.axhline(26.5, linestyle=':', color='r', label='LSST turn-off detection limit')
        plt.axhline(27.5, linestyle='--', color='r', label='LSST limiting mag')

        plt.xlabel('$g - r$ [mag]', fontsize=22)
        plt.ylabel('$g$ [mag]', fontsize=22)
        plt.legend(fontsize=12, loc='upper right')
        plt.xlim(0,1.5)
        plt.ylim(29,15)
        
        plt.tight_layout()
        pp.savefig()
    
    pp.close()
    
    
    
