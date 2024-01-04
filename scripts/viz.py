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

# TNG best-fit potential
bulgePot = gp.HernquistPotential(10**10.26*u.Msun, 1.05*u.kpc, units=galactic)
# issues integrating orbits in a disk potential with vanishing b
#diskPot = gp.MiyamotoNagaiPotential(10**10.70*u.Msun, 4.20*u.kpc, 0.006*u.kpc, units=galactic)
diskPot = gp.MiyamotoNagaiPotential(10**10.70*u.Msun, 4.20*u.kpc, 0.28*u.kpc, units=galactic)
haloPot = gp.NFWPotential(10**11.57*u.Msun, 11.59*u.kpc, c=0.943, units=galactic)

totalPot = gp.CCompositePotential(component1=bulgePot, component2=diskPot, component3=haloPot)
ham = gp.Hamiltonian(totalPot)

coord.galactocentric_frame_defaults.set('v4.0')
gc_frame = coord.Galactocentric()


def get_done(hid=523889, lowmass=True, target='progenitors', verbose=False):
    """Return indices of fully simulated streams"""
    t = Table.read('../data/stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    
    to_run = ind_all
    ind_done = np.zeros(N, dtype=bool)
    
    if target=='progenitors':
        label = 'stream'
    else:
        label = 'gc'
    
    fout = glob.glob('../data/streams/halo.{:d}_{:s}.1.00*'.format(hid, label))
    
    for f in fout:
        i = int(f.split('.')[-2])
        ind_done[i] = 1
    
    if verbose: print(np.sum(ind_done), N)
    
    return ind_done

def plot_streams(hid=523889, lowmass=True, target='progenitors', test=False):
    """Diagnostic plot showing simulated stream particles (downsampled for more massive streams)"""
    t = Table.read('../data/stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    N = len(t)
    
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target)
    
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    
    if test:
        Nplot = 20
    
    plt.close()
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111, projection='mollweide')
    
    for i in range(Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.1.00.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        
        if np.size(cg.l)>1000:
            plt.plot(cg.l.wrap_at(180*u.deg).radian[::10], cg.b.radian[::10], 'o', mew=0, ms=1, alpha=0.1)
        else:
            plt.plot(cg.l.wrap_at(180*u.deg).radian, cg.b.radian, 'o', mew=0, ms=1, alpha=0.1)
    
    plt.tight_layout()


#############
# Summaries #
#############

def estimate_rgal(hid=523889, lowmass=True, target='progenitors'):
    """Calculate Galactocentric radii of stellar streams and store them in the output summary table"""
    t = Table.read('../data/stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    N = len(t)
    
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target)
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    
    t['rgal_stream'] = np.zeros(N) * np.nan * u.kpc
    t['rgal_25'] = np.zeros(N) * np.nan * u.kpc
    t['rgal_75'] = np.zeros(N) * np.nan * u.kpc
    
    for i in range(Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.1.00.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        cgal = cg.transform_to(coord.Galactocentric())
        
        rgal = cgal.spherical.distance.to(u.kpc).value
        
        t['rgal_stream'][to_plot[i]] = np.median(rgal)
        t['rgal_25'][to_plot[i]] = np.percentile(rgal, 25)
        t['rgal_75'][to_plot[i]] = np.percentile(rgal, 75)
    
    t.pprint()
    t.write('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass), overwrite=True)

def estimate_mu(hid=523889, lowmass=True, target='progenitors'):
    """Calculate total surface brightness of stellar streams assuming different healpix binning levels and store them in the output summary table"""
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    N = len(t)
    
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target)
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    
    # surface brightness setup
    wangle = 180*u.deg
    m0 = -48.60
    
    levels = np.array([6,7,8,9,10], dtype=int)
    for level in levels:
        t['mu_{:d}'.format(level)] = np.zeros(N) * np.nan * u.mag * u.arcsec**-2
    
    for i in range(Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.1.00.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        
        for level in levels:
            NSIDE = int(2**level)
            da = (hp.nside2pixarea(NSIDE, degrees=True)*u.deg**2).to(u.arcsec**2)
            
            flux = 10**(-(pkl['g'] - m0)/2.5)
            ind_pix = hp.ang2pix(NSIDE, cg.l.deg, cg.b.deg, nest=False, lonlat=True)
            
            flux_tot = np.zeros(hp.nside2npix(NSIDE))
            
            for e, ind in enumerate(ind_pix):
                flux_tot[ind] += flux[e]
            
            ind_finite = flux_tot>0
            area = np.sum(ind_finite) * da.to(u.arcsec**2)
            mu = -2.5*np.log10(np.sum(flux_tot[ind_finite])/area.to(u.arcsec**2).value) + m0
            
            t['mu_{:d}'.format(level)][to_plot[i]] = mu
    
    t.pprint()
    t.write('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass), overwrite=True)

def estimate_morphology(hid=523889, lowmass=True, target='progenitors'):
    """Calculate median sky positions and orbital properties to determine whether tidal debris is stream-like or phase-mixed"""
    
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    N = len(t)
    
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target)
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    
    t['std_l'] = np.empty(N) * np.nan * u.deg
    t['std_b'] = np.empty(N) * np.nan * u.deg
    t['med_l'] = np.empty(N) * np.nan * u.deg
    t['med_b'] = np.empty(N) * np.nan * u.deg
    
    t['detot'] = np.ones(N) * u.kpc**2*u.Myr**-2
    t['etot'] = np.ones(N) * u.kpc**2*u.Myr**-2
    t['dl'] = np.ones(N) * u.kpc**2*u.Myr**-1
    t['l'] = np.ones(N) * u.kpc**2*u.Myr**-1
    
    for i in range(Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.1.00.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        cgal = cg.transform_to(coord.Galactocentric())
        
        # sky positions
        t['std_l'][to_plot[i]] = np.std(cg.l.wrap_at(180*u.deg).deg)
        t['std_b'][to_plot[i]] = np.std(cg.b.deg)
        t['med_l'][to_plot[i]] = np.median(cg.l.wrap_at(180*u.deg).deg)
        t['med_b'][to_plot[i]] = np.median(cg.b.deg)
        
        # orbital phase space
        w = gd.PhaseSpacePosition(cgal)
        etot = np.abs(w.energy(ham))
        l = np.linalg.norm(w.angular_momentum(), axis=0)
        #l = w.angular_momentum()[2]
        
        t['detot'][to_plot[i]] = np.std(etot.value)
        t['etot'][to_plot[i]] = np.median(etot.value)
        t['dl'][to_plot[i]] = np.std(l.value)
        t['l'][to_plot[i]] = np.median(l.value)
    
    t.pprint()
    t.write('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass), overwrite=True)

def get_streams(hid=523889, lowmass=True, target='progenitors'):
    """"""
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    
    x_ = np.sqrt(t['med_l']**2 + t['med_b']**2)
    y_ = np.abs(t['lz']/t['l'])
    ind_stream = (x_>2) & (y_<0.95)
    
    return ind_stream

##################
# Visualizations #
##################

def rgal_hist(hid=523889, lowmass=True, target='progenitors'):
    """"""
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target)
    ind_stream = get_streams(hid=hid, lowmass=lowmass, target=target)
    
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    
    print('max rgal:', np.max(t['rgal_stream'][ind_done]))
    print('50, 90 rgal:', np.percentile(np.array(t['rgal_stream'][ind_done]), [50,90]))
    
    rbins = np.logspace(-0.8,2.1,100)
    
    plt.close()
    plt.figure(figsize=(10,6))
    
    #plt.hist(t['rgal_25'][ind_done], bins=rbins, histtype='stepfilled', color=mpl.cm.Blues(0.5), alpha=0.2, label='25%')
    #plt.hist(t['rgal_75'][ind_done], bins=rbins, histtype='stepfilled', color=mpl.cm.Blues(0.9), alpha=0.2, label='75%')
    #plt.hist(t['rgal_stream'][ind_done], bins=rbins, histtype='step', color=mpl.cm.Blues(0.7), lw=2, label='Median')
    plt.hist(t['rgal_stream'][ind_done], bins=rbins, histtype='step', color='tab:blue', lw=2, label='All debris')
    plt.hist(t['rgal_stream'][ind_done & ind_stream], bins=rbins, histtype='step', color='tab:red', lw=2, label='Streams')
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    
    plt.legend(fontsize='small', loc=2)
    plt.xlabel('$R_{Gal}$ [kpc]')
    plt.ylabel('Number')
    
    plt.tight_layout()
    plt.savefig('../plots/rgal_total_all.png')

def rgal_mu(hid=523889, lowmass=True, target='progenitors', level=9):
    """"""
    
    tobs = Table.read('../data/gcstreams_mu.csv')
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target)
    ind_stream = get_streams(hid=hid, lowmass=lowmass, target=target)

    plt.close()
    plt.figure(figsize=(10,7))
    
    # model streams
    #plt.plot(t['rgal_stream'], t['mu_7'], 'o', color='skyblue', mew=0, alpha=0.5, label='Predicted streams (level 7)')
    #plt.plot(t['rgal_stream'], t['mu_8'], 'o', color='dodgerblue', mew=0, alpha=0.5, label='Predicted streams (level 8)')
    plt.plot(t['rgal_stream'][ind_done & ind_stream], t['mu_{:d}'.format(level)][ind_done & ind_stream], 'o', color='navy', mew=0, alpha=0.5, label='Predicted GC streams')
    
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
    plt.ylabel('$\mu_{{{:d}}}$ [mag arcsec$^{{-2}}$]'.format(level))
    
    plt.gca().set_xscale('log')
    plt.xlim(1.5,110)
    plt.ylim(39,29)
    
    plt.tight_layout()
    plt.savefig('../plots/rgal_mu_all_{:d}.png'.format(level))

def plot_morphology(hid=523889, lowmass=True, target='progenitors', Nplot=100, log=True):
    """"""
    
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target)
    
    x_ = np.sqrt(t['med_l']**2 + t['med_b']**2)
    y_ = np.abs(t['lz']/t['l'])
    ind_inside = (x_>0.02) & (x_<100) & (y_>0.02) & (y_<0.98)
    
    to_plot = ind_all[ind_done & ind_inside]
    np.random.seed(381)
    to_plot = np.random.choice(to_plot, size=Nplot, replace=False)
    
    t = t[to_plot]
    
    x_ = np.sqrt(t['med_l']**2 + t['med_b']**2)
    y_ = np.abs(t['lz']/t['l'])
    
    plt.close()
    plt.figure(figsize=(12,12))
    
    plt.plot(x_, y_, 'rx', mew=0.5, alpha=0.3, ms=4)
    
    xlim = 2
    ylim = 0.95
    plt.axvline(xlim, ls='--', lw=1.5, color='tab:blue', alpha=0.3)
    plt.axhline(ylim, ls='--', lw=1.5, color='tab:blue', alpha=0.3)
    
    if log:
        plt.gca().set_xscale('log')
    
    plt.xlim(0.015,115)
    plt.ylim(0.01,1)
    
    plt.xlabel('$\Delta_0$ [deg]')
    plt.ylabel('|$L_z$ / L|')
    
    plt.tight_layout()

    # add streams
    ax0 = plt.gca()
    data_bbox = ax0.viewLim
    fig_bbox = ax0.bbox._bbox
    
    xd = x_
    if log:
        kx = (fig_bbox.x1 - fig_bbox.x0) / (np.log10(data_bbox.x1) - np.log10(data_bbox.x0))
        xf = fig_bbox.x0 + kx*(np.log10(xd) - np.log10(data_bbox.x0))
    else:
        kx = (fig_bbox.x1 - fig_bbox.x0) / (data_bbox.x1 - data_bbox.x0)
        xf = fig_bbox.x0 + kx*(xd - data_bbox.x0)

    
    yd = y_
    if log:
        ky = (fig_bbox.y1 - fig_bbox.y0) / (np.log10(data_bbox.y1) - np.log10(data_bbox.y0))
        yf = fig_bbox.y0 + ky*(np.log10(yd) - np.log10(data_bbox.y0))
        
        ky = (fig_bbox.y1 - fig_bbox.y0) / (data_bbox.y1 - data_bbox.y0)
        yf = fig_bbox.y0 + ky*(yd - data_bbox.y0)
    else:
        ky = (fig_bbox.y1 - fig_bbox.y0) / (data_bbox.y1 - data_bbox.y0)
        yf = fig_bbox.y0 + ky*(yd - data_bbox.y0)
    
    aw = 0.1
    ah = 0.5*aw
    
    for i in range(Nplot):
        plt.axes([xf[i]-0.5*aw, yf[i]-0.5*ah, aw, ah], projection='mollweide')
        
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.1.00.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        
        if np.size(cg.l)>10000:
            plt.plot(cg.l.wrap_at(180*u.deg).radian[::2], cg.b.radian[::2], 'ko', ms=0.5, mew=0, alpha=0.2)
        else:
            plt.plot(cg.l.wrap_at(180*u.deg).radian, cg.b.radian, 'ko', ms=0.5, mew=0, alpha=0.2)
        plt.axis('off')
    
    plt.savefig('../plots/streaminess_all_{:d}.png'.format(Nplot))

def plot_distant(hid=523889, lowmass=True, target='progenitors', test=False, rgal=10):
    """Diagnostic plot showing simulated stream particles (downsampled for more massive streams)"""
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    N = len(t)
    
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target)
    
    ind_dist = t['rgal_stream']>rgal
    
    x_ = np.sqrt(t['med_l']**2 + t['med_b']**2)
    y_ = np.abs(t['lz']/t['l'])
    ind_stream = (x_>2) & (y_<0.95)
    
    to_plot = ind_all[ind_done & ind_dist & ind_stream]
    Nplot = np.size(to_plot)
    
    if test:
        Nplot = 20
    
    plt.close()
    fig = plt.figure(figsize=(14,14))
    ax1 = fig.add_subplot(211, projection='mollweide')
    ax2 = fig.add_subplot(212, projection='mollweide')
    
    plt.sca(ax1)
    plt.xlabel('l [deg]')
    plt.ylabel('b [deg]')
    plt.text(0.95, 0.9, 'g<22.5', transform=plt.gca().transAxes)
    
    plt.sca(ax2)
    plt.xlabel('l [deg]')
    plt.ylabel('b [deg]')
    plt.text(0.95, 0.9, 'g<27.5', transform=plt.gca().transAxes)
    
    for i in range(Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.1.00.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        
        ind_sdss = pkl['g']<22.5
        ind_lsst = pkl['g']<27.5
        
        color = 'k'
        
        if np.sum(ind_sdss):
            plt.sca(ax1)
            plt.plot(cg.l.wrap_at(180*u.deg).radian[ind_sdss], cg.b.radian[ind_sdss], 'o', color=color, mew=0, ms=1, alpha=0.1)
        
        if np.sum(ind_lsst):
            plt.sca(ax2)
            plt.plot(cg.l.wrap_at(180*u.deg).radian[ind_lsst], cg.b.radian[ind_lsst], 'o', color=color, mew=0, ms=1, alpha=0.1)
            
    plt.tight_layout()
    plt.savefig('../plots/sdss_lsst_rgal.{:.0f}.png'.format(rgal))

############
# Analysis #
############

def distant_streams(hid=523889, lowmass=True, target='progenitors', dist=10):
    """"""
    
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target)
    
    x_ = np.sqrt(t['med_l']**2 + t['med_b']**2)
    y_ = np.abs(t['lz']/t['l'])
    
    ind_distant = t['rgal_stream']>dist
    ind_stream = (x_>2) & (y_<0.95)
    
    print(np.sum(ind_done), np.sum(ind_done & ind_stream), np.sum(ind_done & ind_distant), np.sum(ind_done & ind_distant & ind_stream))



