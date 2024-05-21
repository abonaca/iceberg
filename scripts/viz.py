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
import tarfile
import os

import healpy as hp
from skimage import io, color

ham_mw = gp.Hamiltonian(gp.MilkyWayPotential())

# TNG best-fit potential 523889
bulgePot1 = gp.HernquistPotential(10**10.26*u.Msun, 1.05*u.kpc, units=galactic)
# issues integrating orbits in a disk potential with vanishing b
#diskPot = gp.MiyamotoNagaiPotential(10**10.70*u.Msun, 4.20*u.kpc, 0.006*u.kpc, units=galactic)
diskPot1 = gp.MiyamotoNagaiPotential(10**10.70*u.Msun, 4.20*u.kpc, 0.28*u.kpc, units=galactic)
haloPot1 = gp.NFWPotential(10**11.57*u.Msun, 11.59*u.kpc, c=0.943, units=galactic)

totalPot1 = gp.CCompositePotential(component1=bulgePot1, component2=diskPot1, component3=haloPot1)
ham1 = gp.Hamiltonian(totalPot1)

# TNG best-fit potential 524506
bulgePot2 = gp.HernquistPotential(10**10.24*u.Msun, 7.07*u.kpc, units=galactic)
diskPot2 = gp.MiyamotoNagaiPotential(10**10.55*u.Msun, 0.5*u.kpc, 0.36*u.kpc, units=galactic)
haloPot2 = gp.NFWPotential(10**11.65*u.Msun, 10.61*u.kpc, c=0.89, units=galactic)

totalPot2 = gp.CCompositePotential(component1=bulgePot2, component2=diskPot2, component3=haloPot2)
ham2 = gp.Hamiltonian(totalPot2)

# TNG best-fit potential 545703
bulgePot3 = gp.HernquistPotential(10**10.38*u.Msun, 8.7*u.kpc, units=galactic)
diskPot3 = gp.MiyamotoNagaiPotential(10**10.34*u.Msun, 0.18*u.kpc, 0.58*u.kpc, units=galactic)
haloPot3 = gp.NFWPotential(10**11.51*u.Msun, 8.41*u.kpc, c=0.942, units=galactic)

totalPot3 = gp.CCompositePotential(component1=bulgePot3, component2=diskPot3, component3=haloPot3)
ham3 = gp.Hamiltonian(totalPot3)

coord.galactocentric_frame_defaults.set('v4.0')
gc_frame = coord.Galactocentric()


def get_done(hid=523889, lowmass=True, target='progenitors', verbose=False, fstar=-1):
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
    
    fout = glob.glob('../data/streams/halo.{:d}_{:s}.{:.2f}*'.format(hid, label, fstar))
    
    for f in fout:
        i = int(f.split('.')[-2])
        ind_done[i] = 1
    
    if verbose: print(np.sum(ind_done), N)
    
    return ind_done

def plot_streams(hid=523889, lowmass=True, target='progenitors', test=False, fstar=-1):
    """Diagnostic plot showing simulated stream particles (downsampled for more massive streams)"""
    t = Table.read('../data/stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    N = len(t)
    
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target, fstar=fstar)
    
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    
    if test:
        Nplot = 20
    
    plt.close()
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(111, projection='mollweide')
    
    pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:.2f}.{:04d}.pkl'.format(hid, fstar, to_plot[-1]), 'rb'))
    # print(to_plot[-1])
    
    for i in range(Nplot):
        print(i, to_plot[i], Nplot, N)
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:.2f}.{:04d}.pkl'.format(hid, fstar, to_plot[i]), 'rb'))
        cg = pkl['cg']
        
        if np.size(cg.l)>1000:
            plt.plot(cg.l.wrap_at(180*u.deg).radian[::20], cg.b.radian[::20], 'o', mew=0, ms=1, alpha=0.1)
        else:
            plt.plot(cg.l.wrap_at(180*u.deg).radian, cg.b.radian, 'o', mew=0, ms=1, alpha=0.1)
    
    plt.tight_layout()
    plt.savefig('../plots/streams_sky_halo.{:d}_fstar.{:.2f}.png'.format(hid, fstar))


#############
# Summaries #
#############

def save_nstar(hid=523889, lowmass=True, target='progenitors', fstar=-1):
    """Query stellar streams and store the total number of stars for each in the output summary table"""
    t = Table.read('../data/stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    N = len(t)
    
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target, fstar=fstar)
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    
    t['nstar'] = np.zeros(N, dtype=np.int64)
    
    for i in range(Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:.2f}.{:04d}.pkl'.format(hid, fstar, to_plot[i]), 'rb'))
        cg = pkl['cg']
        
        t['nstar'][to_plot[i]] = np.size(cg.l)
    
    t.pprint()
    t.write('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.{:.2f}.fits'.format(target, hid, lowmass, fstar), overwrite=True)

def estimate_rgal(hid=523889, lowmass=True, target='progenitors', fstar=-1):
    """Calculate Galactocentric radii of stellar streams and store them in the output summary table"""
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.{:.2f}.fits'.format(target, hid, lowmass, fstar))
    N = len(t)
    
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target, fstar=fstar)
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    
    t['rgal_stream'] = np.zeros(N) * np.nan * u.kpc
    t['rgal_25'] = np.zeros(N) * np.nan * u.kpc
    t['rgal_75'] = np.zeros(N) * np.nan * u.kpc
    
    for i in range(Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:.2f}.{:04d}.pkl'.format(hid, fstar, to_plot[i]), 'rb'))
        cg = pkl['cg']
        cgal = cg.transform_to(coord.Galactocentric())
        
        rgal = cgal.spherical.distance.to(u.kpc).value
        
        t['rgal_stream'][to_plot[i]] = np.median(rgal)
        t['rgal_25'][to_plot[i]] = np.percentile(rgal, 25)
        t['rgal_75'][to_plot[i]] = np.percentile(rgal, 75)
    
    t.pprint()
    t.write('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.{:.2f}.fits'.format(target, hid, lowmass, fstar), overwrite=True)

def estimate_mu(hid=523889, lowmass=True, target='progenitors', fstar=-1):
    """Calculate total surface brightness of stellar streams assuming different healpix binning levels and store them in the output summary table"""
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.{:.2f}.fits'.format(target, hid, lowmass, fstar))
    N = len(t)
    
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target, fstar=fstar)
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    
    # surface brightness setup
    wangle = 180*u.deg
    m0 = -48.60
    
    levels = np.array([6,7,8,9,10], dtype=int)
    for level in levels:
        t['mu_{:d}'.format(level)] = np.zeros(N) * np.nan * u.mag * u.arcsec**-2
    
    for i in range(Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:.2f}.{:04d}.pkl'.format(hid, fstar, to_plot[i]), 'rb'))
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
    t.write('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.{:.2f}.fits'.format(target, hid, lowmass, fstar), overwrite=True)

def estimate_morphology(hid=523889, lowmass=True, target='progenitors', fstar=-1):
    """Calculate median sky positions and orbital properties to determine whether tidal debris is stream-like or phase-mixed"""
    
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.{:.2f}.fits'.format(target, hid, lowmass, fstar))
    N = len(t)
    
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target, fstar=fstar)
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    
    t['std_l'] = np.empty(N) * np.nan * u.deg
    t['std_b'] = np.empty(N) * np.nan * u.deg
    t['med_l'] = np.empty(N) * np.nan * u.deg
    t['med_b'] = np.empty(N) * np.nan * u.deg
    
    t['de'] = np.ones(N) * u.kpc**2*u.Myr**-2
    t['e'] = np.ones(N) * u.kpc**2*u.Myr**-2
    t['dl'] = np.ones(N) * u.kpc**2*u.Myr**-1
    t['l'] = np.ones(N) * u.kpc**2*u.Myr**-1
    
    # switch between halo potentials
    if hid==523889:
        ham = ham1
    elif hid==524506:
        ham = ham2
    elif hid==545703:
        ham = ham3
    else:
        ham = ham_mw
    
    for i in range(Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:.2f}.{:04d}.pkl'.format(hid, fstar, to_plot[i]), 'rb'))
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
        
        t['de'][to_plot[i]] = np.std(etot.value)
        t['e'][to_plot[i]] = np.median(etot.value)
        t['dl'][to_plot[i]] = np.std(l.value)
        t['l'][to_plot[i]] = np.median(l.value)
    
    t.pprint()
    t.write('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.{:.2f}.fits'.format(target, hid, lowmass, fstar), overwrite=True)

def get_streams(hid=523889, lowmass=True, target='progenitors', fstar=-1):
    """Return indices of clusters with stream-like tidal debris"""
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.{:.2f}.fits'.format(target, hid, lowmass, fstar))
    
    x_ = np.sqrt(t['med_l']**2 + t['med_b']**2)
    y_ = np.abs(t['lz']/t['l'])
    ind_stream = (x_>2) & (y_<0.95)
    
    return ind_stream


##################
# Visualizations #
##################

def rgal_hist(hid=523889, lowmass=True, target='progenitors', fstar=-1):
    """Plot histogram of galactocentric radii for the resulting tidal debris, and a subset with stream-like morphologies"""
    
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.{:.2f}.fits'.format(target, hid, lowmass, fstar))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target, fstar=fstar)
    ind_stream = get_streams(hid=hid, lowmass=lowmass, target=target, fstar=fstar)
    
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    
    print('max rgal:', np.max(t['rgal_stream'][ind_done]))
    print('50, 90 rgal:', np.percentile(np.array(t['rgal_stream'][ind_done]), [50,90]))
    
    rbins = np.logspace(-0.8,3.1,100)
    
    plt.close()
    plt.figure(figsize=(10,6))
    
    #plt.hist(t['rgal_25'][ind_done], bins=rbins, histtype='stepfilled', color=mpl.cm.Blues(0.5), alpha=0.2, label='25%')
    #plt.hist(t['rgal_75'][ind_done], bins=rbins, histtype='stepfilled', color=mpl.cm.Blues(0.9), alpha=0.2, label='75%')
    #plt.hist(t['rgal_stream'][ind_done], bins=rbins, histtype='step', color=mpl.cm.Blues(0.7), lw=2, label='Median')
    plt.hist(t['rgal_stream'][ind_done], bins=rbins, histtype='step', color='tab:blue', lw=2, label='All debris ({:d})'.format(np.sum(ind_done)))
    plt.hist(t['rgal_stream'][ind_done & ind_stream], bins=rbins, histtype='step', color='tab:red', lw=2, label='Streams ({:d})'.format(np.sum(ind_done & ind_stream)))
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    
    plt.legend(fontsize='small', loc=2)
    plt.xlabel('$R_{Gal}$ [kpc]')
    plt.ylabel('Number')
    
    plt.tight_layout()
    plt.savefig('../plots/rgal_total_halo.{:d}_fstar.{:.2f}.png'.format(hid, fstar))

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

def plot_morphology(hid=523889, lowmass=True, target='progenitors', Nplot=100, log=True, fstar=-1):
    """"""
    
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.{:.2f}.fits'.format(target, hid, lowmass, fstar))
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
        
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:.2f}.{:04d}.pkl'.format(hid, fstar, to_plot[i]), 'rb'))
        cg = pkl['cg']
        
        if np.size(cg.l)>500:
            plt.plot(cg.l.wrap_at(180*u.deg).radian, cg.b.radian, 'ko', ms=1, mew=0, alpha=0.2)
        else:
            plt.plot(cg.l.wrap_at(180*u.deg).radian, cg.b.radian, 'ko', ms=1, mew=0, alpha=0.8)
        
        # if fstar==1:
        #     if np.size(cg.l)>10000:
        #         plt.plot(cg.l.wrap_at(180*u.deg).radian[::2], cg.b.radian[::2], 'ko', ms=0.5, mew=0, alpha=0.2)
        #     else:
        #         plt.plot(cg.l.wrap_at(180*u.deg).radian, cg.b.radian, 'ko', ms=0.5, mew=0, alpha=0.2)
        plt.axis('off')
    
    plt.savefig('../plots/streaminess_all_{:.2f}_{:d}.png'.format(fstar, Nplot))

def plot_distant(hid=523889, lowmass=True, target='progenitors', test=False, rgal=15):
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

def streams_edgeon(hid=523889, lowmass=True, target='progenitors', test=False, rgal=15):
    """"""
    
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
    
    plt.close()
    plt.figure()
    
    plt.xlabel('X [kpc]')
    plt.ylabel('Z [kpc]')
    
    for i in range(Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.1.00.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        cgal = cg.transform_to(coord.Galactocentric())
        
        #ind_sdss = pkl['g']<22.5
        ind_lsst = pkl['g']<27.5
        
        color = 'k'
        
        if np.sum(ind_lsst):
            plt.plot(cgal.x[ind_lsst], cgal.z[ind_lsst], 'ko', mew=0, ms=1, alpha=0.051)
            #plt.plot(cg.l.wrap_at(180*u.deg).radian[ind_lsst], cg.b.radian[ind_lsst], 'o', color=color, mew=0, ms=1, alpha=0.1)
    
    plt.xlim(-70,70)
    plt.ylim(-70,70)
    plt.gca().set_aspect('equal')
    plt.tight_layout()


############
# Analysis #
############

def nstar_mass(logm, hid=523889, lowmass=True, target='progenitors'):
    """Print a typical number of stars in a cluster of given mass logm"""
    
    fstar = -1
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.{:.2f}.fits'.format(target, hid, lowmass, fstar))

    ind = (t['nstar']>0)
    t = t[ind]
    
    ind = (np.abs(t['logMgc_at_birth']-logm)<0.1)
    
    print(np.mean(t['nstar'][ind]), np.sum(ind))

def distant_streams(hid=523889, lowmass=True, target='progenitors', dist=15):
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

def info(hid=523889, lowmass=True, target='progenitors', dist=15):
    """"""
    
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    print(t.colnames)


def failed_streams(hid=523889, lowmass=True, target='progenitors', fstar=-1):
    """Look into which streams failed to run"""
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.{:.2f}.fits'.format(target, hid, lowmass, fstar))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target, fstar=fstar)
    
    # print(t.colnames)
    # print(np.sum(~ind_done))
    
    bins_l = np.linspace(0,1,100)
    bins_rp = np.linspace(0,3,100)
    
    plt.close()
    fig, ax = plt.subplots(1,3,figsize=(15,5))
    
    plt.sca(ax[0])
    plt.plot(t['lz'], t['etot'], 'ko', label='All ({:d})'.format(N))
    plt.plot(t['lz'][~ind_done], t['etot'][~ind_done], 'ro', label='Failed ({:d})'.format(np.sum(~ind_done)))
    
    plt.legend(fontsize='small')
    plt.xlabel('$L_z$ [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('$E_{tot}$ [kpc$^2$ Myr$^{-2}$]')
    
    plt.sca(ax[1])
    plt.hist(np.abs(t['lz']), bins=bins_l, histtype='step', color='k', lw=2, alpha=0.8, label='$r_{{all,max}}$ = {:.2f} kpc$^2$Myr$^{{-1}}$'.format(np.max(np.abs(t['lz']))))
    plt.hist(np.abs(t['lz'][~ind_done]), bins=bins_l, histtype='step', color='r', lw=2, alpha=0.8, label='$r_{{failed,max}}$ = {:.2f} kpc$^2$Myr$^{{-1}}$'.format(np.max(np.abs(t['lz'][~ind_done]))))
    
    plt.legend(fontsize='small')
    plt.xlabel('|$L_z$| [kpc$^2$ Myr$^{-1}$]')
    plt.ylabel('Number')
    
    plt.sca(ax[2])
    plt.hist(np.abs(t['rperi']), bins=bins_rp, histtype='step', color='k', lw=2, alpha=0.8, label='$r_{{all,max}}$ = {:.1f} kpc'.format(np.max(t['rperi'])))
    plt.hist(np.abs(t['rperi'][~ind_done]), bins=bins_rp, histtype='step', color='r', lw=2, alpha=0.8, label='$r_{{failed,max}}$ = {:.1f} kpc'.format(np.max(t['rperi'][~ind_done])))
    
    plt.legend(fontsize='small')
    plt.xlabel('$r_{peri}$ [kpc]')
    plt.ylabel('Number')
    
    plt.tight_layout()
    plt.savefig('../plots/failed_diagnostics_halo.{:d}_fstar.{:.2f}.png'.format(hid, fstar))

def done(hid=523889, lowmass=True, target='progenitors', fstar=-1):
    """"""
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.{:.2f}.fits'.format(target, hid, lowmass, fstar))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target, fstar=fstar)
    ind_stream = get_streams(hid=hid, lowmass=lowmass, target=target, fstar=-1)
    
    istream = ind_all[ind_stream]
    idone = ind_all[ind_done]
    
    imissing = np.array([0 if x in idone else 1 for x in istream], dtype=bool)
    print('Missing:', istream[imissing])
    print(np.sum(imissing))
    
    np.save('../data/to_run_halo.{:d}_{:s}.{:.2f}.npy'.format(hid, target, fstar), ind_all[istream[imissing]])
    
    print(np.size(ind_all[istream[imissing]]), np.size(istream))
    
    #print(istream)
    #print(imissing)

def tar_full(hid=523889, lowmass=True, target='progenitors'):
    """Create a tarball of full-run morphological streams"""
    fstar = -1
    t = Table.read('../data/output_stream_{:s}_halo.{:d}_lowmass.{:d}_fstar.{:.2f}.fits'.format(target, hid, lowmass, fstar))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = get_done(hid=hid, lowmass=lowmass, target=target, fstar=fstar)
    ind_stream = get_streams(hid=hid, lowmass=lowmass, target=target, fstar=fstar)
    
    to_store = ind_all[ind_done & ind_stream]
    Nstore = np.size(to_store)
    
    fnames = ['/home/ana/projects/iceberg/data/streams/halo.{:d}_stream.1.00.{:04d}.pkl'.format(hid, x) for x in to_store if os.path.isfile('/home/ana/projects/iceberg/data/streams/halo.{:d}_stream.1.00.{:04d}.pkl'.format(hid, x))]
    
    print(fnames, len(fnames), Nstore)
    
    tarball_name = '../data/streams/full_streams_halo.{:d}.tar.gz'.format(hid)
    
    # Create a tarball and open it for writing with gzip compression
    with tarfile.open(tarball_name, 'w:gz') as tar:
        for file in fnames:
            # Add each file to the tarball
            tar.add(file, arcname=file)


