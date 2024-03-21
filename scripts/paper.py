import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

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
import healpy as hp

ham_mw = gp.Hamiltonian(gp.MilkyWayPotential())

bulgePot = ham_mw.potential['bulge']
diskPot = ham_mw.potential['disk']
haloPot = ham_mw.potential['halo']
totalPot = gp.CCompositePotential(component1=bulgePot, component2=diskPot, component3=haloPot)
ham = gp.Hamiltonian(totalPot)

coord.galactocentric_frame_defaults.set('v4.0')
gc_frame = coord.Galactocentric()

from mock import *

def plot_halo_streams(hid=523889, lowmass=True, glim=27):
    """"""
    
    t = Table.read('../data/stream_progenitors_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    
    ind_halo = get_halo(t, full=False)
    ind_all = np.arange(N, dtype=int)
    to_run = ind_all[ind_halo]
    print(np.size(to_run))
    
    plt.close()
    fig = plt.figure(figsize=(13,6))
    ax = fig.add_subplot(111, projection='mollweide')
    plt.xlabel('Galactic longitude [deg]')
    plt.ylabel('Galactic latitude [deg]')
    plt.title('g<{:.1f}'.format(glim), fontsize='medium')
    
    wangle = 180*u.deg
    dmin = 1*u.kpc
    dmax = 50*u.kpc
    
    for i in to_run[:]:
        try:
            pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, i), 'rb'))
            cg = pkl['cg']
            ind_lim = pkl['g']<glim
            
            plt.scatter(cg.l.wrap_at(wangle).rad[ind_lim], cg.b.rad[ind_lim], c=cg.distance.value[ind_lim], cmap='plasma', vmin=dmin.value, vmax=dmax.value, s=0.1, alpha=0.1, rasterized=True)
            
        except FileNotFoundError:
            pass
    
    # add custom colorbar
    sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=dmin.value, vmax=dmax.value))
    # fake up the array of the scalar mappable. Urgh...
    sm._A = []
    
    plt.colorbar(sm, label='Distance [kpc]', ax=ax, pad=0.04, aspect=20)

    #plt.axis('off')
    plt.tight_layout()
    plt.savefig('../plots/paper/halo_streams_distance_glim.{:.1f}.pdf'.format(glim), dpi=200)

def rgal_mu(level):
    """"""
    
    tobs = Table.read('../data/streams_mu.csv')
    t = Table.read('../data/streams_sb_halo.523889_lowmass.1.fits')
    
    plt.close()
    plt.figure(figsize=(12,7))
    
    # model streams
    #plt.plot(t['rgal_stream'], t['mu_7'], 'o', color='skyblue', mew=0, alpha=0.5, label='Predicted streams (level 7)')
    #plt.plot(t['rgal_stream'], t['mu_8'], 'o', color='dodgerblue', mew=0, alpha=0.5, label='Predicted streams (level 8)')
    plt.plot(t['rgal_stream'], t['mu_{:d}'.format(level)], 'o', color='navy', mew=0, alpha=0.5, label='Predicted GC streams')
    
    # observed streams
    plt.plot(tobs['Rgal'], tobs['mu'], '*', ms=15, c='gold', mec='k', label='Milky Way streams')
    
    for e, name in enumerate(tobs['Stream']):
        if name in ['Turbio', 'Molonglo', 'Aliqa Uma', 'Turranburra']:
            dy = -0.13
            dx = 0.5
        else:
            dy = 0.1
            dx = 0.7
        plt.text(tobs['Rgal'][e]+dx, tobs['mu'][e]+dy, name, fontsize='xx-small', va='center')
    
    plt.legend(frameon=True, fontsize='small')
    #plt.xlabel('Galactocentric radius [kpc]')
    #plt.ylabel('Surface brightness [mag arcsec$^{-2}$]')
    plt.xlabel('$R_{Gal}$ [kpc]')
    plt.ylabel('$\mu_{:d}$ [mag arcsec$^{{-2}}$]'.format(level))
    
    #plt.gca().set_xscale('log')
    plt.xlim(0,100)
    plt.ylim(37,31)
    
    plt.tight_layout()
    plt.savefig('../plots/rgal_mu_{:d}.png'.format(level))
