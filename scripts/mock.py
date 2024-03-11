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

#bulgePot = ham_mw.potential['bulge']
#diskPot = ham_mw.potential['disk']
#haloPot = ham_mw.potential['halo']
#totalPot = gp.CCompositePotential(component1=bulgePot, component2=diskPot, component3=haloPot)
#ham = gp.Hamiltonian(totalPot)

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


def read_table():
    """"""
    t = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    t.pprint()
    
def t_disrupt(i=0):
    """"""
    
    t = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    hid = np.unique(t['haloID'])
    print(hid)
    ind = t['haloID'] == hid[i]
    t = t[ind]
    
    r = np.sqrt(t['x']**2 + t['y']**2 + t['z']**2)

    print(len(t), np.sum(r>10))
    
    plt.close()
    fig, ax = plt.subplots(3,1,figsize=(10,10), sharex=True)
    
    plt.sca(ax[0])
    plt.plot(t['t_disrupt'], t['logMgc_at_birth'], 'ko')
    plt.ylabel('log $M_{gc}/M_\odot$')
    
    plt.sca(ax[1])
    plt.plot(t['t_disrupt'], t['rperi'], 'ko')
    plt.ylabel('$R_{peri}$ [kpc]')
    
    plt.sca(ax[2])
    plt.plot(t['t_disrupt'], r, 'ko')
    plt.gca().set_yscale('log')
    plt.xlabel('$T_{disrupt}$ [Gyr]')
    plt.ylabel('$R_{gal}$ [kpc]')
    
    plt.tight_layout()
    plt.savefig('../plots/tdisrupt_halo.{:d}.png'.format(hid[i]))

def masses():
    """"""
    t = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    tgc = Table.read('../data/mw_like_4.0_0.5_lambda_survive.txt', format='ascii.commented_header', delimiter=' ')
    hid = np.unique(t['haloID'])
    
    bins = np.linspace(4,7,30)
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(9.5,5), sharex=True, sharey=True)
    
    for i in range(3):
        plt.sca(ax[0])
        ind = t['haloID'] == hid[i]
        plt.hist(t['logMgc_at_birth'][ind], bins=bins, density=True, histtype='step', label='{:d}'.format(hid[i]))

        plt.sca(ax[1])
        ind = tgc['haloID'] == hid[i]
        plt.hist(tgc['logMgc_at_birth'][ind], bins=bins, density=True, histtype='step', label='{:d}'.format(hid[i]))
    
    plt.sca(ax[0])
    plt.xlabel('log $M_{gc}/M_\odot$')
    plt.ylabel('Density')
    plt.title('Disrupted', fontsize='medium')
    
    plt.sca(ax[1])
    plt.legend(loc=1, fontsize='small')
    plt.xlabel('log $M_{gc}/M_\odot$')
    #plt.ylabel('Density')
    plt.title('Surviving', fontsize='medium')
    
    plt.tight_layout()
    plt.savefig('../plots/masses.png')

def orbits(i=0, test=True):
    """"""
    t = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    #t = t[:50]
    hid = np.unique(t['haloID'])
    ind = t['haloID'] == hid[i]
    t = t[ind]
    if test:
        t = t[::20]
    
    c = coord.Galactocentric(x=t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    N = 5000
    dt = 0.5*u.Myr
    orbit = ham.integrate_orbit(w0, dt=dt, n_steps=N)
    rperi = orbit.pericenter()
    rapo = orbit.apocenter()
    
    x = np.linspace(0,100)
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    plt.sca(ax[0])
    plt.plot(x, x, 'r-')
    plt.plot(rperi, t['rperi'], 'ko')
    
    plt.xlim(0,20)
    plt.ylim(0,20)
    plt.xlabel('$R_{peri}$ [kpc]')
    plt.ylabel('$R_{peri, TNG}$ [kpc]')
    
    plt.sca(ax[1])
    plt.plot(x, x, 'r-')
    plt.plot(rapo, t['rapo'], 'ko')
    
    plt.xlim(0,60)
    plt.ylim(0,60)
    plt.xlabel('$R_{apo}$ [kpc]')
    plt.ylabel('$R_{apo, TNG}$ [kpc]')
    
    plt.tight_layout()
    plt.savefig('../plots/orbit_comparison_halo.{:d}.png'.format(hid[i]))


def stream_origin():
    """"""
    
    tall = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    hid = np.unique(tall['haloID'])
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(12,6))
    
    for i in range(3):
        ind = tall['haloID'] == hid[i]
        t = tall[ind]
        c = coord.Galactocentric(x=t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s)
        
        ind_insitu = t['t_accrete']==-1
        ind_acc = ~ind_insitu
        
        Nd = 50
        dist = np.linspace(1,50,Nd) * u.kpc
        facc = np.empty(Nd)
        fis = np.empty(Nd)
        for j in range(Nd):
            ind_dist = c.spherical.distance<dist[j]
            facc[j] = np.sum(ind_dist & ind_acc) / np.sum(ind_dist)
            
            ind_dist = c.spherical.distance>dist[j]
            fis[j] = np.sum(ind_dist & ind_insitu) / np.sum(ind_dist)
            
        plt.sca(ax[0])
        plt.plot(dist, facc, '-', label='{:d}'.format(hid[i]))
    
        plt.sca(ax[1])
        plt.plot(dist, fis, '-')
    
    plt.sca(ax[0])
    plt.xlabel('$R_{Gal}$ [kpc]')
    plt.ylabel('$f_{acc}(<R_{Gal})$')
    plt.legend(loc=4, fontsize='small')

    plt.sca(ax[1])
    plt.xlabel('$R_{Gal}$ [kpc]')
    plt.ylabel('$f_{in\,situ}(>R_{Gal})$')

    plt.tight_layout()
    plt.savefig('../plots/stream_origin.png')

def test_stream():
    """"""
    tall = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    hid = np.unique(tall['haloID'])
    ind = tall['haloID']==hid[0]
    t = tall[ind]
    
    ind_insitu = (t['t_accrete']==-1)
    t = t[ind_insitu]
    t.pprint()
    ind = np.argmax(t['rperi'])
    t = t[ind]
    
    c = coord.Galactocentric(x=t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s)
    #c = coord.SkyCoord(ra=tc['RAdeg'], dec=tc['DEdeg'], distance=tc['Dist'], pm_ra_cosdec=tc['pmRA'], pm_dec=tc['pmDE'], radial_velocity=tc['HRV'], frame='icrs')
    g = c.transform_to(coord.Galactic)
    rep = c.transform_to(coord.Galactocentric).data
    w0 = gd.PhaseSpacePosition(rep)
    
    gc_mass = 10**t['logMgc_at_birth']*u.Msun
    rh = 10*u.pc
    tform = t['t_form']*u.Gyr
    
    rs = np.random.RandomState(5134)
    df = ms.FardalStreamDF(random_state=rs)
    
    gc_pot = gp.PlummerPotential(m=gc_mass, b=rh, units=galactic)
    mw = gp.MilkyWayPotential()
    
    dt = -1*u.Myr
    n_steps = np.int(np.abs((tform/dt).decompose()))
    nskip = 100
    
    gen_stream = ms.MockStreamGenerator(df, mw, progenitor_potential=gc_pot)
    stream, _ = gen_stream.run(w0, gc_mass, dt=dt, n_steps=n_steps, release_every=nskip)
    

# realistic potential

def potential_info(hid=506151):
    """Print properties of the analytical potential representing the TNG halo"""
    
    diskPot = agama.Potential('../data/pot_disk_{:d}.pot'.format(hid))
    haloPot = agama.Potential('../data/pot_halo_{:d}.pot'.format(hid))
    totalPot = agama.Potential(diskPot, haloPot)

    print('{:s} potential for the halo; value at origin = {:f} (km/s)^2'.format(haloPot.name(), haloPot.potential(0,0,0)))
    print('{:s} potential for the halo; value at origin = {:f} (km/s)^2'.format(diskPot.name(), diskPot.potential(0,0,0)))
    print('Total potential for the halo; value at origin = {:f} (km/s)^2'.format(totalPot.potential(0,0,0)))
    
def test_orbit(hid=506151, test=True):
    """Test orbit evolution in the agama potential"""
    
    t = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    ind = t['haloID'] == hid
    t = t[ind]
    if test:
        t = t[::20]
    
    # present-day positions of clusters = initial positions for orbit calculation
    c = coord.Galactocentric(x=t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s)
    ic = np.array([c.x.value, c.y.value, c.z.value, c.v_x.value, c.v_y.value, c.v_z.value]).T
    
    # gravitational potential
    diskPot = agama.Potential('../data/pot_disk_{:d}.pot'.format(hid))
    haloPot = agama.Potential('../data/pot_halo_{:d}.pot'.format(hid))
    totalPot = agama.Potential(diskPot, haloPot)

    # evolve
    Nstep = 300
    timeinit = -13  # 3 Gyr back in time
    timecurr =  0.0  # current time is 0, the final point
    times, orbits = agama.orbit(ic=ic, potential=totalPot, timestart=timecurr, time=timeinit-timecurr, trajsize=Nstep).T
    
    # organize output in a single 3D numpy array
    orbit = np.empty((len(orbits),Nstep,6))
    for i in range(len(orbits)):
        orbit[i] = orbits[i]
    
    # 
    r = np.sqrt(orbit[:,:,0]**2 + orbit[:,:,1]**2 + orbit[:,:,2]**2)
    rperi = np.min(r, axis=1)
    rapo = np.max(r, axis=1)
    
    # plotting
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(10,5))
    
    plt.sca(ax[0])
    plt.plot(orbit[:,0,0], orbit[:,0,2], 'k.')
    plt.plot(orbit[:,-1,0], orbit[:,-1,2], 'ro')
    
    plt.gca().set_aspect('equal')
    plt.xlabel('X [kpc]')
    plt.ylabel('Z [kpc]')
    
    plt.sca(ax[1])
    plt.plot(rperi, t['rperi'], 'k.')
    plt.plot(rapo, t['rapo'], 'r.')
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.gca().set_aspect('equal')
    plt.xlabel('Local estimate')
    plt.ylabel('Official estimate')
    
    plt.tight_layout()


def approx_halo(hid=506151):
    """Find NFW model in gala that reproduces the 0,0 radial trend for agama multipole halo model"""
    
    t = Table.read('../data/pot_halo_{:d}.pot'.format(hid), header_start=1, data_start=6, data_end=26, format='ascii.commented_header', delimiter='\t')
    t.pprint()
    
    nfw = gp.NFWPotential(m=4.75e11*u.Msun, r_s=14*u.kpc, units=galactic)
    nfw = gp.NFWPotential(m=4.4e11*u.Msun, r_s=13*u.kpc, units=galactic)
    r = np.logspace(-1.5,5.3,100) * u.pc
    c = coord.Galactocentric(x=np.zeros_like(r), y=np.zeros_like(r), z=r)
    
    plt.close()
    plt.figure()
    
    plt.plot(t['radius'], t['l=0,m=0'], 'ko')
    #plt.plot(t['radius'], t['l=2,m=0'], 'bo')
    
    plt.plot(r, nfw.energy([c.x, c.y, c.z]).to(u.km**2*u.s**-2), 'r-')
    
    plt.gca().set_xscale('log')
    plt.tight_layout()
    
def test_orbit_gala(hid=506151, test=True):
    """"""
    t = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    ind = t['haloID'] == hid
    t = t[ind]
    if test:
        t = t[::20]
    
    # define potentials
    diskPot = gp.CylSplinePotential.from_file('../data/pot_disk_{:d}.pot'.format(hid), units=galactic)
    # placeholder until learning how to define agama-style multipole in gala
    haloPot = gp.NFWPotential(m=4.4e11*u.Msun, r_s=13*u.kpc, units=galactic)
    totalPot = gp.CompositePotential(component1=diskPot, component2=haloPot)
    
    # present-day positions of clusters = initial positions for orbit calculation
    c = coord.Galactocentric(x=t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # integrate orbits
    N = 5000
    dt = 1*u.Myr
    orbit = totalPot.integrate_orbit(w0, dt=dt, n_steps=N)
    rperi = orbit.pericenter()
    rapo = orbit.apocenter()

    x = np.linspace(0,100)
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(11,5))
    
    plt.sca(ax[0])
    plt.plot(x, x, 'r-')
    plt.plot(rperi, t['rperi'], 'ko')
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.xlabel('$R_{peri}$ [kpc]')
    plt.ylabel('$R_{peri, TNG}$ [kpc]')
    
    plt.sca(ax[1])
    plt.plot(x, x, 'r-')
    plt.plot(rapo, t['rapo'], 'ko')
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.xlabel('$R_{apo}$ [kpc]')
    plt.ylabel('$R_{apo, TNG}$ [kpc]')
    
    plt.tight_layout()
    plt.savefig('../plots/orbit_comparison_tngpot_halo.{:d}.png'.format(hid))


def elz(hid=0):
    """Stream progenitors in the ELz phase space"""
    
    t = Table.read('../data/mw_like_6.0_0.4_2.5_linear_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    hid = np.unique(t['haloID'])[hid]
    ind = (t['haloID'] == hid) & ((t['t_accrete']==-1) | (t['t_disrupt']<t['t_accrete']))
    t = t[ind]
    N = len(t)
    
    c = coord.Galactocentric(x=-1*t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=-1*t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # integration set up
    haloPot = gp.NFWPotential(m=4.4e11*u.Msun, r_s=13*u.kpc, units=galactic)
    totalPot = haloPot
    #ham = gp.Hamiltonian(totalPot)
    dt = -1*u.Myr
    
    orbit = ham.integrate_orbit(w0, dt=0.1*u.Myr, n_steps=0)
    
    etot = orbit.energy()[0].reshape(N,-1)
    lz = orbit.angular_momentum()[2][0].reshape(N,-1)
    
    print(np.sum(np.abs(lz.value)>0.3))
    print(np.sum(etot.value>-0.2))
    print(np.sum(((np.abs(lz.value)>0.3) | (etot.value>-0.21))))
    print(N)
    
    plt.close()
    plt.figure()
    
    plt.plot(lz, etot, 'k.', ms=1)
    plt.axvline(0.3)
    plt.axvline(-0.3)
    plt.axhline(-0.21)
    #plt.hist(np.abs(lz.value), bins=np.linspace(0,2,100))
    
    plt.tight_layout()


def stream_number(hid=0, lowmass=True):
    """Print stream number for each halo"""
    
    if lowmass:
        t = Table.read('../data/mw_like_6.0_0.4_2.5_linear_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    else:
        t = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    print('Total number in 3 halos: {:d}'.format(len(t)))
    
    for hid in np.unique(t['haloID']):
        ind = (t['haloID']==hid) & ((t['t_accrete']==-1) | (t['t_disrupt']<t['t_accrete']))
        print('Halo: {:d}, streams: {:d}, disrupted in the main halo: {:d}'.format(hid, np.sum(t['haloID']==hid), np.sum(ind)))

def all_orbits(hid=0, lowmass=True, test=False, disrupt=True):
    """Calculate orbits of all disrupted globular clusters in a halo"""
    
    if disrupt:
        label_in = 'disrupt'
        label = 'progenitors'
    else:
        label_in = 'survive'
        label = 'gcs'
    
    # read table
    if lowmass:
        t = Table.read('../data/mw_like_6.0_0.4_2.5_linear_{:s}.txt'.format(label_in), format='ascii.commented_header', delimiter=' ')
    else:
        t = Table.read('../data/mw_like_4.0_0.5_lambda_{:s}.txt'.format(label_in), format='ascii.commented_header', delimiter=' ')
    
    hid = np.unique(t['haloID'])[hid]
    ind = (t['haloID']==hid) & ((t['t_accrete']==-1) | (t['t_disrupt']<t['t_accrete']))
    t = t[ind]
    
    if test:
        t = t[:10]
    
    N = len(t)
    
    # setup coordinates
    c = coord.Galactocentric(x=-1*t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=-1*t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # integration setup
    dt = -1*u.Myr
    T = 5*u.Gyr
    nstep = np.abs(int((T/dt).decompose()))
    
    # integrate orbits
    orbit = ham.integrate_orbit(w0, dt=dt, n_steps=nstep)
    pickle.dump(orbit, open('../data/orbits_{:s}_halo.{:d}_lowmass.{:d}.pkl'.format(label, hid, lowmass), 'wb'))
    
    # orbit summary
    rperi = orbit.pericenter()
    rapo = orbit.apocenter()
    etot = orbit.energy()[0].reshape(N,-1)[:,0]
    lz = orbit.angular_momentum()[2][0].reshape(N,-1)[:,0]
    
    t['rperi_mw'] = rperi
    t['rapo_mw'] = rapo
    t['etot'] = etot
    t['lz'] = lz
    
    # so we can save in fits format with no complaints
    t.rename_column('[Fe/H]', 'FeH')
    
    t.pprint()
    t.write('../data/stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(label, hid, lowmass), overwrite=True)
    
def compare_orbits(hid=523889, lowmass=True, fraction=True):
    """Compare orbital peri- and apocenters in the MW potential vs TNG"""
    
    # read table
    t = Table.read('../data/stream_progenitors_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    
    if fraction:
        res_peri = 1 - t['rperi_mw']/t['rperi']
        res_apo = 1 - t['rapo_mw']/t['rapo']
        ylabels = ['1 - $R_p$ / $R_{p,0}$', '1 - $R_a$ / $R_{a,0}$']
        ylims = [[-1,1], [-1,1]]
    else:
        res_peri = t['rperi'] - t['rperi_mw']
        res_apo = t['rapo'] - t['rapo_mw']
        ylabels = ['$R_{p,0}$ - $R_p$', '$R_{a,0}$ - $R_a$']
        ylims = [[-2,2], [-5,5]]
    
    med_peri = np.median(res_peri)
    std_peri = np.std(res_peri)
    
    med_apo = np.median(res_apo)
    std_apo = np.std(res_apo)
    
    
    plt.close()
    fig, ax = plt.subplots(1,2,figsize=(15,5))
    
    plt.sca(ax[0])
    plt.axhline(0, color='r')
    plt.plot(t['rperi_mw'], res_peri, 'ko', mew=0, ms=2)
    
    plt.ylim(ylims[0])
    plt.gca().set_xscale('log')
    plt.text(0.1, 0.9, 'Median: {:.2f}\nSTD: {:.2f}'.format(med_peri, std_peri), fontsize='small', transform=plt.gca().transAxes, va='top')
    
    plt.xlabel('$R_p$ [kpc]')
    plt.ylabel(ylabels[0])
    
    plt.sca(ax[1])
    plt.axhline(0, color='r')
    plt.plot(t['rapo_mw'], res_apo, 'ko', mew=0, ms=2)
    
    plt.ylim(ylims[1])
    plt.gca().set_xscale('log')
    plt.text(0.1, 0.9, 'Median: {:.2f}\nSTD: {:.2f}'.format(med_apo, std_apo), fontsize='small', transform=plt.gca().transAxes, va='top')
    
    plt.xlabel('$R_a$ [kpc]')
    plt.ylabel(ylabels[1])
    
    plt.tight_layout()
    plt.savefig('../plots/orbit_comparison_halo.{:d}_f.{:d}.png'.format(hid, fraction))



def pot(R):
    return ham.potential.energy([R, 0, 0]*u.kpc).value[0]

pot_vec = np.vectorize(pot)

def Lcirc(Etot, R):
    return -R*((2*(Etot - pot_vec(R)))**0.5) 

def maxLcirc(Etot):
    optfunc = partial(Lcirc,Etot)
    res = minimize(optfunc, np.array([0.1]), method='BFGS')
    return np.abs(res.fun)

def circularity(hid=523889, lowmass=True, target='progenitors'):
    """"""
    
    t = Table.read('../data/stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    
    # calculate circularity
    maxLcirc_vec = np.vectorize(maxLcirc)
    maxLcirc_arr = maxLcirc_vec(np.linspace(-0.265, 0, 1000))
    
    t['lmax'] = (np.interp(t['etot'], np.linspace(-0.265, 0, 1000), maxLcirc_arr))
    t['circLz'] = np.abs(t['lz']/t['lmax'])
    
    t.pprint()
    t.write('../data/stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass), overwrite=True)


def host_massloss(hid=523889, lowmass=True, target='progenitors'):
    """Estimate mass loss in the progenitor dwarf galaxy"""
    
    t = Table.read('../data/stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    ind_insitu = t['t_accrete']==-1
    
    f_dyn = np.ones(len(t))
    f_dyn[~ind_insitu] = 1 - 0.06*(t['t_form'][~ind_insitu] - t['t_accrete'][~ind_insitu])
    
    t['f_dyn'] = f_dyn
    
    t.pprint()
    t.write('../data/stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass), overwrite=True)

def max_mass(hid=523889, lowmass=True, target='progenitors', verbose=True):
    """Construct a dictionary with a maximum mass of a living star for a given isochrone"""
    
    t = Table.read('../data/stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))

    # isochrones
    age = np.around(t['t_form'], decimals=1)
    feh = np.around(t['FeH'], decimals=1)
    
    label = np.array(['age{:.1f}feh{:.1f}'.format(age_, feh_) for age_, feh_ in zip(age, feh)])
    
    # construct dictionary
    mmax = dict()
    
    for l in np.unique(label)[:]:
        if verbose: print(l)
        age = float(l.split('age')[1].split('feh')[0])
        feh = float(l.split('feh')[1])
        if verbose: print(age, feh)
        
        iso_lsst = read_isochrone(age=age*u.Gyr, feh=feh, ret=True, facility='lsst')
        mmax[l] = np.max(iso_lsst['initial_mass'])
    
    print(mmax)
    pickle.dump(mmax, open('../data/max_mass_{:s}_halo.{:d}_lowmass.{:d}.pkl'.format(target, hid, lowmass), 'wb'))

def nstar_sample(hid=523889, lowmass=True, target='progenitors', test=False):
    """"""
    
    if target=='progenitors':
        # fully dissolved
        label = 'stream'
    else:
        # surviving cluster
        label = 'gc'
    
    t = Table.read('../data/stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    N = len(t)
    
    # starting time of dissolution (birth for in-situ, accretion time for accreted)
    # caveat: missing fluff of stars potentially dissolved before accretion time
    ind_insitu = t['t_accrete']==-1
    t_start = t['t_accrete'][:]
    t_start[ind_insitu] = t['t_form'][ind_insitu][:]
    
    t_end = t['t_disrupt'][:]
    
    prog_mass = 10**t['logMgc_at_birth']*u.Msun
    
    t_prog = t['t_form'] - t['t_accrete']
    f_dyn = 1 - 0.06*(t_prog)
    insitu_mass = 10**t['logMgc_at_birth']*u.Msun
    insitu_mass[~ind_insitu] = (1 - f_dyn[~ind_insitu]) * prog_mass[~ind_insitu]
    
    # define number of steps to start of dissolution
    dt = -1*u.Myr
    #print((t_start*u.Gyr/np.abs(dt)).decompose())
    n_steps = np.int64((t_start*u.Gyr/np.abs(dt)).decompose())
    n_disrupted = np.int64((t_end*u.Gyr/np.abs(dt)).decompose())
    n_disrupting = n_steps - n_disrupted
    
    # isochrones
    age = np.around(t['t_form'], decimals=1)
    feh = np.around(t['FeH'], decimals=1)
    iso_label = np.array(['age{:.1f}feh{:.1f}'.format(age_, feh_) for age_, feh_ in zip(age, feh)])
    mmax = pickle.load(open('../data/max_mass_{:s}_halo.{:d}_lowmass.{:d}.pkl'.format(target, hid, lowmass), 'rb'))
    
    sampled_mass = np.zeros_like(prog_mass)
    nstar = np.zeros(N, dtype=int)
    
    if test:
        N = 100
    
    for i in range(N):
        mass_stream = prog_mass[i].value
        
        masses = sample_kroupa(np.log10(mass_stream))
        sampled_mass[i] = np.sum(masses) * u.Msun
        
        # pre-accretion mass loss
        if ~ind_insitu[i]:
            np.random.seed(52)
            np.random.shuffle(masses)
            
            cumsum_mass = np.cumsum(masses)
            ind_mass = np.argmin(np.abs(cumsum_mass - insitu_mass[i].value))
            masses = masses[:ind_mass+1]
            
        # don't simulate stellar remnants
        ind_alive = masses<mmax[iso_label[i]]
        masses = masses[ind_alive]
        
        nstar[i] = np.size(masses)
    
    delta_mass = prog_mass - sampled_mass
    ntail = np.int64(nstar/2)
    navg = np.array(ntail / n_disrupting)
    
    print(np.percentile(navg, [0.1,1,5,50,99]), np.min(navg))
    print(np.sum(navg<1))
    print(np.sum(~ind_insitu))
    print(np.sum((navg<1) & ind_insitu))
    print(np.percentile(ntail, [0.1,1,5,50,99]), np.min(ntail))
    
    plt.close()
    fig, ax = plt.subplots(1,4,figsize=(16,4))
    
    plt.sca(ax[0])
    plt.hist(delta_mass.value, bins=100)
    plt.axvline(np.median(delta_mass.value), color='r')
    plt.xlabel('$\Delta$M [$M_\odot$]')
    plt.ylabel('Number')
    
    plt.sca(ax[1])
    plt.hist(delta_mass.value/prog_mass.value, bins=100)
    plt.xlabel('$\Delta$M / $M_{gc}$')
    plt.ylabel('Number')
    
    plt.sca(ax[2])
    plt.hist(nstar, bins=np.logspace(2,6.5))
    plt.gca().set_xscale('log')
    plt.xlabel('N$_\star$')
    plt.ylabel('Number')
    
    plt.sca(ax[3])
    plt.hist(navg, bins=np.logspace(-2,4.1))
    plt.axvline(np.median(navg), color='r')
    plt.gca().set_xscale('log')
    plt.xlabel('N$_{release}$')
    plt.ylabel('Number')
    
    plt.tight_layout()
    plt.savefig('../plots/sampling_diagnostics.png')


def get_halo(t, full=True):
    """"""
    
    ind_apo = (t['rapo_mw']>5)
    #ind_apo = (t['rapo']>5)
    ind_peri = (t['rperi_mw']>5)
    ind_disk = (t['circLz']>0.5) & (t['lz']<0)
    
    ind_halo = ~ind_disk & ind_apo & ind_peri
    ind_halo = ~ind_disk & ind_apo
    
    if full:
        ind_halo = ind_apo
    else:
        ind_halo = ~ind_disk & ind_apo & ind_peri
    
    return ind_halo

def define_halo(hid=523889, lowmass=True):
    """"""
    
    t = Table.read('../data/stream_progenitors_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    
    ind_halo = get_halo(t)
    print(np.sum(ind_halo), len(t))
    
    plt.close()
    plt.figure()
    
    plt.plot(t['lz'], t['etot'], 'ko', mew=0, ms=2)
    plt.plot(t['lz'][ind_halo], t['etot'][ind_halo], 'ro', mew=0, ms=2)
    
    plt.tight_layout()


def mock_stream(hid=523889, test=True, graph=True, i0=0, f=0.3, Nmax=1000, lowmass=True, target='progenitors', verbose=False, halo=True, istart=0, nskip=1, remaining=False):
    """Create a mock stream from a disrupted globular cluster"""
    
    if target=='progenitors':
        # fully dissolved
        label = 'stream'
    else:
        # surviving cluster
        label = 'gc'
    
    
    t = Table.read('../data/stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    N = len(t)
    
    # starting time of dissolution (birth for in-situ, accretion time for accreted)
    # caveat: missing fluff of stars potentially dissolved before accretion time
    ind_insitu = t['t_accrete']==-1
    t_start = t['t_accrete'][:]
    t_start[ind_insitu] = t['t_form'][ind_insitu][:]
    
    t_end = t['t_disrupt'][:]
    
    # cluster mass
    prog_mass = 10**t['logMgc_at_birth']*u.Msun
    t_prog = t['t_form'] - t['t_accrete']
    f_dyn = 1 - 0.06*(t_prog)

    insitu_mass = 10**t['logMgc_at_birth']*u.Msun
    insitu_mass[~ind_insitu] = (1 - f_dyn[~ind_insitu]) * prog_mass[~ind_insitu]
    
    
    # present-day cluster positions
    c = coord.Galactocentric(x=-1*t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=-1*t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # integration set up
    dt = -1*u.Myr
    dt = -0.1*u.Myr
    dt = -0.5*u.Myr
    
    # setup how stars get released from the progenitor
    state = np.random.RandomState(seed=291)
    df = ms.FardalStreamDF(random_state=state)
    gen = ms.MockStreamGenerator(df, ham)
    
    ind_all = np.arange(N, dtype=int)
    if halo:
        if target=='progenitors':
            ind_halo = get_halo(t)
        else:
            ind_halo = get_halo(t, full=True)
        to_run = ind_all[ind_halo]
    else:
        to_run = np.arange(i0, N, 1, dtype=int)
    
    if remaining:
        to_run = np.load('../data/to_run_{:s}.npy'.format(target))
    
        if test:
            to_run = [to_run[i0],]
    
    print(np.size(to_run))
    #to_run = to_run[::-1]
    
    if test:
        #to_run = [to_run[0],]
        to_run = [ind_all[i0],]
    
    for i in to_run[istart::nskip]:
        try:
            if verbose: print('gc {:d}, logM = {:.2f}'.format(i, t['logMgc_at_birth'][i]))
            
            # define number of steps to start of dissolution
            n_steps = int((t_start[i]*u.Gyr/np.abs(dt)).decompose())
            n_disrupted = int((t_end[i]*u.Gyr/np.abs(dt)).decompose())
            n_disrupting = n_steps - n_disrupted
            
            # still surviving globular clusters
            if t_end[i]<0:
                n_disrupting = n_steps
                n_disrupted = 0
            
            # read isochrones
            age = np.around(t['t_form'][i], decimals=1)
            feh = np.around(t['FeH'][i], decimals=1)
            iso_lsst = read_isochrone(age=age*u.Gyr, feh=feh, ret=True, facility='lsst')
            
            # total mass of stream stars
            mass_stream = prog_mass[i].value
            
            # for surviving clusters, only simulate released stars
            if t_end[i]<0:
                # remaining mass
                current_mass = 10**t['logMgc'][i]
                mass_stream = mass_stream - current_mass
            
            if mass_stream<=0:
                print('All stream stars released prior to accretion onto the Milky Way')
            else:
                # sample masses
                masses = sample_kroupa(np.log10(mass_stream))
                np.random.seed(25)
                np.random.shuffle(masses)
                
                # for accreted clusters, account for stars tidally stripped in the original host galaxy
                if ~ind_insitu[i]:
                    cumsum_mass = np.cumsum(masses)
                    ind_mass = np.argmin(np.abs(cumsum_mass - insitu_mass[i].value))
                    masses = masses[:ind_mass+1]
                
                # subsample mass
                if (f<1) & (f>0):
                    nchoose = int(f * np.size(masses))
                    np.random.seed(91)
                    masses = np.random.choice(masses, size=nchoose, replace=False)
                
                ## sort masses (assuming lowest mass get kicked out first)
                #masses = np.sort(masses)
                
                # don't simulate stellar remnants
                bbox = [np.min(iso_lsst['initial_mass']), np.max(iso_lsst['initial_mass'])]
                ind_alive = masses<bbox[1]
                masses = masses[ind_alive]
                
                # make it an even number of stars
                if np.size(masses)%2:
                    masses = masses[:-1]
                
                nstar = np.size(masses)
                
                if f<0:
                    ntail = min(Nmax, n_steps+1)
                    nrelease = np.zeros(n_steps+1, dtype=int)
                    navg = n_steps//ntail
                    nrelease[::navg] = 1
                    ntail = np.sum(nrelease)
                    
                    masses = np.partition(masses, -2*ntail)[-2*ntail:]
                    release_every = navg
                    nrelease = 1
                
                else:
                    # uniformly release the exact number of stars
                    release_every = 1
                    ntail = int(nstar/2)
                    navg = ntail//n_disrupting
                    if navg>=1:
                        nrelease = np.ones(n_disrupting, dtype=int) * navg
                        nextra = int(ntail - np.sum(nrelease))
                        nrelease[:nextra] += 1
                    else:
                        # fewer particles than time steps
                        nrelease = np.zeros(n_disrupting, dtype=int)
                        release_avg = int(np.ceil(n_disrupting/ntail))
                        nrelease[::release_avg] = 1

                        nextra = int(ntail - np.sum(nrelease))
                        # get integer locations rather than a boolean array, so that we can slice it and assign a new value
                        ind_empty = np.where(nrelease==0)[0]
                        nrelease[ind_empty[:nextra]] = 1
                    
                    # add extra steps
                    nrelease = np.concatenate([nrelease, np.zeros(n_steps - n_disrupting + 1, dtype=int)])
                
                if verbose: print('Ntail: {:d}, Nrelease: {:d}'.format(ntail, np.sum(nrelease)))

                # create mock stream
                t1 = time.time()
                stream_, prog = gen.run(w0[i], prog_mass[i], dt=dt, n_steps=n_steps, n_particles=nrelease, release_every=release_every)
                t2 = time.time()
                if verbose: print('Runtime: {:g}'.format(t2-t1))
                
                ##################
                # paint photometry
                # get distance modulus
                cg = stream_.to_coord_frame(coord.Galactic())
                dm = 5*np.log10((cg.distance.to(u.pc)).value) - 5
                
                # interpolate isochrone
                order = 1
                interp_g = InterpolatedUnivariateSpline(iso_lsst['initial_mass'], iso_lsst['LSST_g'], k=order, bbox=bbox)
                interp_r = InterpolatedUnivariateSpline(iso_lsst['initial_mass'], iso_lsst['LSST_r'], k=order, bbox=bbox)
                
                # photometry (no uncertainties)
                r = interp_r(masses) + dm
                g = interp_g(masses) + dm
                
                #############
                # save stream
                outdict = dict(cg=cg, mass=masses, g=g, r=r)
                pickle.dump(outdict, open('../data/streams/halo.{:d}_{:s}.{:.2f}.{:04d}.pkl'.format(hid, label, f, i), 'wb'))
                
                if graph:
                    # plot sky coordinates
                    ind = np.arange(np.size(cg.l))
                    
                    plt.close()
                    plt.figure()
                    
                    #plt.plot(cg.l, cg.b, 'k.')
                    plt.scatter(cg.l.wrap_at(180*u.deg), cg.b, c=ind, s=cg.distance.value**-1)
                    
                    plt.xlabel('l [deg]')
                    plt.ylabel('b [deg]')
                    plt.gca().set_aspect('equal')
                    plt.tight_layout()
                    plt.savefig('../plots/streams/halo.{:d}_{:s}.{:04d}.png'.format(hid, label, i))
        except Exception as error:
            print('An exception occurred at stream {:d}: '.format(i), error)
            fout_failed = open('failed_ids', 'a')
            fout_failed.write('{:d}\n'.format(i))
            fout_failed.close()

def get_remaining(hid=523889, lowmass=True, halo=True, target='progenitors'):
    """"""
    t = Table.read('../data/stream_{:s}_halo.{:d}_lowmass.{:d}.fits'.format(target, hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    
    if halo:
        ind_halo = get_halo(t)
        to_run = ind_all[ind_halo]
    else:
        to_run = ind_all
    
    if target=='progenitors':
        label = 'stream'
    else:
        label = 'gc'
    
    fout = glob.glob('../data/streams/halo.{:d}_{:s}.1.00*'.format(hid, label))
    print(len(fout), len(to_run))
    
    ind_remaining = np.ones(N, dtype=bool)
    if halo:
        ind_remaining[~ind_halo] = 0
    
    for f in fout:
        i = int(f.split('.')[-2])
        ind_remaining[i] = 0
    
    print(np.sum(ind_remaining), len(to_run)-len(fout))
    print(ind_all[ind_remaining].tolist())

    np.save('../data/to_run_{:s}.npy'.format(target), ind_all[ind_remaining])
    
    


def nstar():
    """"""
    m1 = 0.1
    m2 = 0.5
    m3 = 15
    alpha = 1.3
    beta = 2.3
    
    int_n1 = (m2**(1-alpha) - m1**(1-alpha))/(1-alpha)
    int_n2 = (m3**(1-beta) - m2**(1-beta))/(1-beta)
    int_n = int_n1 + int_n2
    
    #int_n = (m2**(1-alpha) - m1**(1-alpha))/(1-alpha) + (m3**(1-beta) - m2**(1-beta))/(1-beta)
    int_m = (m2**(2-alpha) - m1**(2-alpha))/(2-alpha) + (m3**(2-beta) - m2**(2-beta))/(2-beta)
    
    print(int_n/int_m*10**6)
    print(int_n/int_m*10**5)
    print(int_n/int_m*10**4)


def sample_kroupa(logm, seed=13, graph=False, ret=True):
    """Sample the Kroupa IMF to get masses of stars in a cluster with a total mass of 10**logm
    Following: https://mathworld.wolfram.com/RandomNumber.html"""
    
    # Kroupa IMF parameters and mass limits of 0.1Msun and 15Msun
    m1 = 0.1
    m2 = 0.5
    m3 = 15
    alpha = 1.3
    beta = 2.3
    
    # evaluate relevant integrals
    int_n1 = (m2**(1-alpha) - m1**(1-alpha))/(1-alpha)
    int_n2 = (m3**(1-beta) - m2**(1-beta))/(1-beta)
    
    int_m = (m2**(2-alpha) - m1**(2-alpha))/(2-alpha) + (m3**(2-beta) - m2**(2-beta))/(2-beta)*m2**(beta-alpha)
    
    x1 = 10**logm / int_m
    x2 = m2**(beta-alpha)*x1
    
    # number of stars in each component
    N1 = int(x1*int_n1)
    N2 = int(x2*int_n2)
    
    # normalizations for sampling the powerlaws
    C1 = (1-alpha)/(m2**(1-alpha) - m1**(1-alpha))
    C2 = (1-beta)/(m3**(1-beta) - m2**(1-beta))
    
    # sample powerlaws
    np.random.seed(seed)
    y1 = np.random.random(N1)
    y2 = np.random.random(N2)
    
    masses_1 = ((1-alpha)/C1*y1 + m1**(1-alpha))**(1/(1-alpha))
    masses_2 = ((1-beta)/C2*y2 + m2**(1-beta))**(1/(1-beta))
    
    masses = np.concatenate([masses_1, masses_2])
    
    if graph:
        plt.close()
        plt.hist(masses, bins=np.logspace(-1,1.3,100), density=True)

        plt.gca().set_xscale('log')
        plt.gca().set_yscale('log')
        plt.tight_layout()
        
    if ret:
        return masses

def paint_masses(hid=0):
    """"""
    
    t = Table.read('../data/mw_like_6.0_0.4_2.5_linear_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    hid = np.unique(t['haloID'])[hid]
    ind = (t['haloID'] == hid) & ((t['t_accrete']==-1) | (t['t_disrupt']<t['t_accrete']))
    t = t[ind]
    
    # load stream
    i = 0
    stream = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, i), 'rb'))
    N = np.size(stream.x)
    
    masses = sample_kroupa(t['logMgc_at_birth'][i])
    #print(np.log10(np.sum(masses)), t['logMgc_at_birth'][i], np.size(masses))

    ind_alive = masses<0.8052
    ind_alive = masses<1.2
    masses = masses[ind_alive]
    print(np.log10(np.sum(masses)), t['logMgc_at_birth'][i], np.size(masses), N)

    Nstar = np.size(masses)
    
    if N<Nstar:
        p = (1 - N/Nstar) * 100
        print(p, np.percentile(masses,p))
        stream_masses = np.random.choice(masses, size=N, replace=False)
    else:
        stream_masses = masses
    
    plt.close()
    plt.figure()
    
    plt.hist(masses, bins=np.logspace(-1,1.3,100), alpha=0.3, density=True, label='Full cluster')
    plt.hist(stream_masses, bins=np.logspace(-1,1.3,100), color='tab:blue', lw=2, density=True, histtype='step', label='Stream samples')
    
    plt.xlabel('Mass [$M_\odot$]')
    plt.ylabel('Density [$M_\odot^{-1}$]')
    plt.legend()
    
    plt.xlim(0.1, 1)
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.tight_layout()

def needed_isochrones(hid=0, target='disrupt'):
    """Print ages and metallicities needed"""
    
    t = Table.read('../data/mw_like_6.0_0.4_2.5_linear_{:s}.txt'.format(target), format='ascii.commented_header', delimiter=' ')
    hid = np.unique(t['haloID'])[hid]
    ind = (t['haloID'] == hid) & ((t['t_accrete']==-1) | (t['t_disrupt']<t['t_accrete']))
    t = t[ind]
    
    age = np.around(t['t_form'], decimals=1)
    feh = np.around(t['[Fe/H]'], decimals=1)
    
    iso = np.vstack([age, feh])
    #print(iso)
    #print(np.unique(iso, axis=1))
    #print(np.shape(iso), np.shape(np.unique(iso, axis=1)))
    print(np.shape(np.unique(feh)))
    print(np.shape(np.unique(age)))
    print(np.unique(age)*1e9)
    print(np.unique(feh))

def read_isochrone(age=10.2*u.Gyr, feh=-2.5, graph=False, verbose=False, ret=True, facility='lsst'):
    """Return isochrone of a given age and metallicity (extracted from a joint file of a given metallicity)
    graph=True - plots interpolation into all LSST bands"""
    
    # read isochrone
    iso_full = Table.read('../data/isochrones/{:s}/mist_gc_{:.1f}.cmd'.format(facility, feh), format='ascii.commented_header', header_start=12)
    ind_age = np.around(iso_full['isochrone_age_yr'], decimals=1)==age.to(u.yr).value
    iso = iso_full[ind_age]
    if verbose: iso.pprint()
    
    if graph:
        bands = ['u', 'g', 'r', 'i', 'z', 'y']
        
        plt.close()
        fig, ax = plt.subplots(6,1,figsize=(12,10), sharex=True)
        
        for e, b in enumerate(bands):
            plt.sca(ax[e])
            
            bbox = [np.min(iso['initial_mass']), np.max(iso['initial_mass'])]
            order = 1
            interp = InterpolatedUnivariateSpline(iso['initial_mass'], iso['LSST_{:s}'.format(b)], k=order, bbox=bbox)
            
            x_ = np.linspace(bbox[0], bbox[1], 1000000)
            y_ = interp(x_)
            
            plt.plot(x_, y_, '-', label='Interpolation')
            plt.plot(iso['initial_mass'], iso['LSST_{:s}'.format(b)], 'o', label='Isochrone points')
            
            plt.ylabel('LSST_{:s}'.format(b))
        
        plt.legend()
        plt.xlabel('Initial mass [$M_\odot$]')
        
        plt.tight_layout(h_pad=0)
        plt.savefig('../plots/sample_isochrone.png')
    
    if ret:
        return iso

def paint_photometry(hid=0, i=0, seed=141):
    """"""
    
    #t = Table.read('../data/mw_like_6.0_0.4_2.5_linear_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    t = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    hid = np.unique(t['haloID'])[hid]
    ind = (t['haloID'] == hid) & ((t['t_accrete']==-1) | (t['t_disrupt']<t['t_accrete']))
    t = t[ind]
    
    # load stream
    #stream = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, i), 'rb'))
    stream = pickle.load(open('../data/streams/halo.{:d}_stream.{:03d}.pkl'.format(hid, i), 'rb'))
    N = np.size(stream.x)
    
    # read isochrone
    age = np.around(t['t_form'][i], decimals=1)
    feh = np.around(t['[Fe/H]'][i], decimals=1)
    iso = read_isochrone(age=age*u.Gyr, feh=feh, ret=True)
    
    # interpolate isochrone
    bbox = [np.min(iso['initial_mass']), np.max(iso['initial_mass'])]
    order = 1
    interp_g = InterpolatedUnivariateSpline(iso['initial_mass'], iso['LSST_g'], k=order, bbox=bbox)
    interp_r = InterpolatedUnivariateSpline(iso['initial_mass'], iso['LSST_r'], k=order, bbox=bbox)
    
    # sample masses
    masses = sample_kroupa(t['logMgc_at_birth'][i])
    ind_alive = masses<bbox[1]
    masses = masses[ind_alive]
    
    # subsample if necessary
    np.random.seed(seed)
    Nstar = np.size(masses)
    if N<Nstar:
        # model underdense > subsample masses
        stream_masses = np.random.choice(masses, size=N, replace=False)
    else:
        # model overdense > subsample stream points
        stream_masses = masses
        
        ind_ = np.arange(N, dtype=int)
        ind_sub = np.random.choice(ind_, size=Nstar, replace=False)
        stream = stream[ind_sub]
        print(np.shape(stream))
    
    # distances
    ceq = stream.to_coord_frame(coord.ICRS())
    dist = ceq.distance
    dm = 5*np.log10((dist.to(u.pc)).value)-5
    
    # photometry (no uncertainties)
    r = interp_r(stream_masses) + dm
    g = interp_g(stream_masses) + dm
    
    # diagnostic plot
    cg = stream.to_coord_frame(coord.Galactic())
    glim = 22
    ind_visible = g<glim
    wangle = 180*u.deg
    
    plt.close()
    fig, ax = plt.subplots(1,2, gridspec_kw=dict(width_ratios=(1,3)), figsize=(12,4))
    
    plt.sca(ax[0])
    plt.plot(g-r, g, 'ko', ms=1, alpha=0.1)
    plt.axhline(glim, color='r')
    
    plt.gca().invert_yaxis()
    plt.xlabel('g - r')
    plt.ylabel('g')
    
    plt.sca(ax[1])
    plt.plot(cg.l.wrap_at(wangle), cg.b, 'ko', ms=1, label='All')
    plt.plot(cg.l.wrap_at(wangle)[ind_visible], cg.b[ind_visible], 'ro', ms=1, label='g<{:.1f}'.format(glim))
    
    plt.gca().set_aspect('equal')
    plt.xlabel('l [deg]')
    plt.ylabel('b [deg]')
    plt.legend(markerscale=4, handlelength=0.5, fontsize='small', loc=1)
    
    plt.tight_layout()
    plt.savefig('../plots/mock_photometry_{:d}.{:d}.png'.format(hid, i))

def save_photometry(hid=0, full=False, verbose=True, seed=139):
    """"""
    if full:
        t = Table.read('../data/mw_like_6.0_0.4_2.5_linear_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    else:
        t = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    hid = np.unique(t['haloID'])[hid]
    ind = (t['haloID'] == hid) & ((t['t_accrete']==-1) | (t['t_disrupt']<t['t_accrete']))
    t = t[ind]
    Nstream = len(t)
    
    for i in range(Nstream):
        if verbose: print('{:d} / {:d}'.format(i, Nstream))
        # load stream
        if full:
            stream = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, i), 'rb'))
        else:
            stream = pickle.load(open('../data/streams/halo.{:d}_stream.{:03d}.pkl'.format(hid, i), 'rb'))
        N = np.size(stream.x)
        
        # read isochrone
        age = np.around(t['t_form'][i], decimals=1)
        feh = np.around(t['[Fe/H]'][i], decimals=1)
        iso = read_isochrone(age=age*u.Gyr, feh=feh, ret=True)
        
        # interpolate isochrone
        bbox = [np.min(iso['initial_mass']), np.max(iso['initial_mass'])]
        order = 1
        interp_g = InterpolatedUnivariateSpline(iso['initial_mass'], iso['LSST_g'], k=order, bbox=bbox)
        interp_r = InterpolatedUnivariateSpline(iso['initial_mass'], iso['LSST_r'], k=order, bbox=bbox)
        
        # sample masses
        masses = sample_kroupa(t['logMgc_at_birth'][i])
        ind_alive = masses<bbox[1]
        masses = masses[ind_alive]
        
        # subsample if necessary
        np.random.seed(seed)
        Nstar = np.size(masses)
        if N<Nstar:
            # model underdense > subsample masses
            stream_masses = np.random.choice(masses, size=N, replace=False)
        else:
            # model overdense > subsample stream points
            stream_masses = masses
            
            ind_ = np.arange(N, dtype=int)
            ind_sub = np.random.choice(ind_, size=Nstar, replace=False)
            stream = stream[ind_sub]
        
        # distances
        ceq = stream.to_coord_frame(coord.ICRS())
        cg = stream.to_coord_frame(coord.Galactic())
        dist = ceq.distance
        dm = 5*np.log10((dist.to(u.pc)).value)-5
        
        # photometry (no uncertainties)
        r = interp_r(stream_masses) + dm
        g = interp_g(stream_masses) + dm
        
        # output
        dictout = dict(stream=cg, r=r, g=g, mass=stream_masses)
        pickle.dump(dictout, open('../data/streams/phot_halo.{:d}_stream.{:04d}.pkl'.format(hid, i), 'wb'))


# All streams

def plot_survey(glim=22, test=False, full=False, hid=0, halo=False):
    """Plot all streams in galactic sky coordinates down to a magnitude limit"""
    
    if full:
        t = Table.read('../data/mw_like_6.0_0.4_2.5_linear_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    else:
        t = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    hid = np.unique(t['haloID'])[hid]
    ind = (t['haloID'] == hid) & ((t['t_accrete']==-1) | (t['t_disrupt']<t['t_accrete']))
    t = t[ind]
    
    if test:
        Nstream = 10
    else:
        Nstream = len(t)
    
    wangle = 180*u.deg
    nskip = 2
    
    # setup plot
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(12,6), subplot_kw=dict(projection='mollweide'))
    
    for i in range(Nstream):
        # load stream
        pkl = pickle.load(open('../data/streams/phot_halo.{:d}_stream.{:04d}.pkl'.format(hid, i), 'rb'))
        stream = pkl['stream']
        ind = pkl['g']<glim
        
        if halo & (t['rapo'][i]<5):
            pass
        else:
            im = plt.scatter(stream.l.wrap_at(wangle).rad[ind][::nskip], stream.b.rad[ind][::nskip], s=5, c=stream.distance.value[ind][::nskip], ec='none', alpha=0.2, cmap='magma', vmin=0, vmax=40)
    
    plt.colorbar(label='Distance [kpc]')
    #plt.grid()
    #plt.axis('off')
    plt.xlabel('l [deg]')
    plt.xlabel('b [deg]')
    
    plt.tight_layout()
    plt.savefig('../plots/streams_sky_halo.{:d}_glim.{:.1f}_halo.{:d}.png'.format(hid, glim, halo))

def plot_sky(hid=506151, test=True, colorby='mass'):
    """"""
    t = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    ind = t['haloID'] == hid
    t = t[ind]
    #print(t.colnames)
    #print(np.size(np.unique(t['origin_leaf'])))
    
    if test:
        N = 100
    else:
        N = len(t)
    
    wangle = 180*u.deg
    
    if colorby=='mass':
        label = 'logMgc_at_birth'
    else:
        label = 't_form'
        colorby = 'age'
    
    cmin = np.min(t[label])
    dc = np.max(t[label]) - cmin
    
    #lzmin = np.min(t['Jp'])
    #dlz = np.max(t['Jp']) - lzmin
    #print(lzmin, dlz)
    
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(12,6.5), subplot_kw=dict(projection='mollweide'), facecolor='k')
    
    for i in range(N):
        stream_ = pickle.load(open('../data/streams/halo.{:d}_stream.{:03d}.pkl'.format(hid, i), 'rb'))
        stream = stream_.to_coord_frame(coord.Galactic())[::10]
        
        cind = np.abs(t[label][i]-cmin)/dc
        
        plt.scatter(stream.l.wrap_at(wangle).rad, stream.b.rad, s=stream.distance*0.1, color=mpl.cm.jet(cind), alpha=0.1, edgecolors='none')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../plots/streams_sky_halo.{:d}_{:s}.png'.format(hid, colorby), facecolor=fig.get_facecolor())

def plot_origin(hid=506151, k=0):
    """"""
    
    t = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    ind = t['haloID'] == hid
    t = t[ind]
    
    mmin = np.min(t['logMgc_at_birth'])
    dm = np.max(t['logMgc_at_birth']) - mmin
    
    leaves =np.unique(t['origin_leaf'])
    #print(leaves)
    Nleaf = np.size(leaves)
    if k>=Nleaf:
        k = Nleaf - 1
    
    ind = t['origin_leaf']==leaves[k]
    t = t[ind]
    N = len(t)
    
    print(leaves[k], N)
    
    wangle = 180*u.deg
    
    #lzmin = np.min(t['Jp'])
    #dlz = np.max(t['Jp']) - lzmin
    #print(lzmin, dlz)
    
    plt.close()
    fig, ax = plt.subplots(1,1,figsize=(12,6.5), subplot_kw=dict(projection='mollweide'), facecolor='k')
    
    for i in range(N):
        stream_ = pickle.load(open('../data/streams/halo.{:d}_stream.{:03d}.pkl'.format(hid, i), 'rb'))
        stream = stream_.to_coord_frame(coord.Galactic())
        
        if N>10:
            stream = stream[::10]
        
        cind = np.abs(t['logMgc_at_birth'][i]-mmin)/dm
        
        plt.scatter(stream.l.wrap_at(wangle).rad, stream.b.rad, s=stream.distance*0.1, color=mpl.cm.jet(cind), alpha=0.1, edgecolors='none')
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../plots/streams_sky_halo.{:d}_leaf.{:d}.png'.format(hid, k), facecolor=fig.get_facecolor())


# Halo streams

def apocenters(hid=0, lowmass=True):
    """Plot histogram of apocenters"""
    
    if lowmass:
        t = Table.read('../data/mw_like_6.0_0.4_2.5_linear_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    else:
        t = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    
    hid = np.unique(t['haloID'])[hid]
    ind = (t['haloID'] == hid) & ((t['t_accrete']==-1) | (t['t_disrupt']<t['t_accrete']))
    t = t[ind]
    
    # halo
    ind_halo = t['rperi']>5
    print(np.sum(ind_halo), len(t))
    
    plt.close()
    plt.figure()
    
    plt.hist(t['rapo'], bins=np.linspace(0,20,100), alpha=0.3)
    
    plt.tight_layout()

def halo_sky(hid=523889, lowmass=True, glim=22, nskip=1):
    """"""
    
    t = Table.read('../data/stream_progenitors_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    
    ind_halo = get_halo(t)
    ind_all = np.arange(N, dtype=int)
    ind_halo = get_halo(t)
    to_run = ind_all[ind_halo]

    plt.close()
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111, projection='aitoff', label='polar')
    
    for i in to_run[:]:
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, i), 'rb'))
        cg = pkl['cg']
        
        ind_lim = pkl['g']<glim
        
        plt.scatter(cg.l.wrap_at(180*u.deg).radian[ind_lim][::nskip], cg.b.radian[ind_lim][::nskip], c=cg.distance.value[ind_lim][::nskip], norm=mpl.colors.LogNorm(vmin=5, vmax=70), s=2, alpha=0.1, linewidths=0)
    
    plt.xlabel('l [deg]')
    plt.ylabel('b [deg]')
    plt.title('g < {:g}'.format(glim), fontsize='medium')
    
    plt.tight_layout()
    plt.savefig('../plots/sky_halostreams_g.{:.1f}_nskip.{:d}.png'.format(glim, nskip))

def plot_single(hid=523889, i=0, mag_lim=27, band='r', nskip=1):
    """"""
    pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, i), 'rb'))
    cg = pkl['cg']
    
    dm = 5*np.log10((cg.distance.to(u.pc)).value) - 5
    dist = 12*u.Mpc
    dm_eg = 5*np.log10((dist.to(u.pc)).value) - 5
    
    ind_lim = pkl['{:s}'.format(band)] - dm + dm_eg < mag_lim
    print(np.sum(ind_lim), np.size(ind_lim))
    
    plt.close()
    plt.figure()
    
    plt.plot(cg.l.wrap_at(180*u.deg).deg[ind_lim][::nskip], cg.b.deg[ind_lim][::nskip], 'ko')
    
    plt.xlabel('l [deg]')
    plt.ylabel('b [deg]')
    plt.gca().set_aspect('equal')
    
    plt.tight_layout()

def get_streaminess(hid=523889, lowmass=True, halo=True):
    """"""
    t = Table.read('../data/stream_progenitors_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    
    if halo:
        ind_halo = get_halo(t)
        to_run = ind_all[ind_halo]
    else:
        to_run = ind_all
    
    fout = glob.glob('../data/streams/halo.{:d}*'.format(hid))
    print(len(fout), len(to_run))
    
    ind_remaining = np.ones(N, dtype=bool)
    ind_remaining[~ind_halo] = 0
    
    for f in fout:
        i = int(f.split('.')[-2])
        ind_remaining[i] = 0
    
    ind_done = ind_halo & ~ind_remaining
    
    t['detot'] = np.ones(N) * u.kpc**2*u.Myr**-2
    t['etot'] = np.ones(N) * u.kpc**2*u.Myr**-2
    t['dl'] = np.ones(N) * u.kpc**2*u.Myr**-1
    t['l'] = np.ones(N) * u.kpc**2*u.Myr**-1
    
    keys = ['detot', 'etot', 'dl', 'l']
    for k in keys:
        t[k][~ind_done] = np.nan
    
    for i_ in range(np.sum(ind_done)):
        i = ind_all[ind_done][i_]
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, i), 'rb'))
        cg = pkl['cg']
        cgal = cg.transform_to(coord.Galactocentric())
        w = gd.PhaseSpacePosition(cgal)
        
        etot = np.abs(w.energy(ham))
        l = np.linalg.norm(w.angular_momentum(), axis=0)
        #l = w.angular_momentum()[2]
        
        t['detot'][i] = np.std(etot.value)
        t['etot'][i] = np.median(etot.value)
        t['dl'][i] = np.std(l.value)
        t['l'][i] = np.median(l.value)
    
    t.write('../data/streams_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass), overwrite=True)
    
    plt.close()
    plt.figure()
    
    plt.plot(t['dl']/t['l'], t['detot']/t['etot'], 'ko', ms=4)
    
    plt.xlabel('$\sigma$|L| / |L|')
    plt.ylabel('$\sigma$E$_{tot}$ / E$_{tot}$')
    
    plt.tight_layout()

def plot_morphology_elz(hid=523889, lowmass=True, halo=True, Nplot=100):
    """"""
    
    t = Table.read('../data/streams_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = np.isfinite(t['l']) & np.isfinite(t['dl']) #& (t['dl']/t['l']>1e-2)
    
    to_plot = ind_all[ind_done]
    np.random.seed(381)
    to_plot = np.random.choice(to_plot, size=Nplot, replace=False)
    
    t = t[to_plot]
    
    plt.close()
    plt.figure(figsize=(12,12))
    
    plt.plot(t['l'], t['etot'], 'rx', ms=3, mew=0.2, alpha=0.5)
    
    #plt.gca().set_xscale('log')
    #plt.gca().set_yscale('log')
    
    plt.xlabel('$L$')
    plt.ylabel('E$_{tot}$')
    
    plt.tight_layout()

    # add streams
    ax0 = plt.gca()
    data_bbox = ax0.viewLim
    fig_bbox = ax0.bbox._bbox
    
    xd = t['lz']
    kx = (fig_bbox.x1 - fig_bbox.x0) / (np.log10(data_bbox.x1) - np.log10(data_bbox.x0))
    xf = fig_bbox.x0 + kx*(np.log10(xd) - np.log10(data_bbox.x0))
    
    yd = t['etot']
    ky = (fig_bbox.y1 - fig_bbox.y0) / (np.log10(data_bbox.y1) - np.log10(data_bbox.y0))
    yf = fig_bbox.y0 + ky*(np.log10(yd) - np.log10(data_bbox.y0))
    
    aw = 0.06
    ah = 0.5*aw
    
    for i in range(Nplot):
        plt.axes([xf[i]-0.5*aw, yf[i]-0.5*ah, aw, ah], projection='mollweide')
        
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        
        plt.plot(cg.l.wrap_at(180*u.deg).radian[::100], cg.b.radian[::100], 'ko', ms=0.5, mew=0, alpha=0.2)
        plt.axis('off')
    
    plt.savefig('../plots/streaminess_elz_{:d}.png'.format(Nplot))


def plot_morphology_old(hid=523889, lowmass=True, halo=True, Nplot=100):
    """"""
    
    t = Table.read('../data/streams_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = np.isfinite(t['l']) & (t['dl']/t['l']>1e-2)
    
    to_plot = ind_all[ind_done]
    np.random.seed(381)
    to_plot = np.random.choice(to_plot, size=Nplot, replace=False)
    
    t = t[to_plot]
    
    plt.close()
    plt.figure(figsize=(12,12))
    
    plt.plot(t['dl']/t['l'], t['detot']/t['etot'], 'rx', ms=3, mew=0.2, alpha=0.5)
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    
    plt.xlabel('$\sigma$|L| / |L|')
    plt.ylabel('$\sigma$E$_{tot}$ / E$_{tot}$')
    
    plt.tight_layout()

    # add streams
    ax0 = plt.gca()
    data_bbox = ax0.viewLim
    fig_bbox = ax0.bbox._bbox
    
    xd = t['dl']/t['l']
    kx = (fig_bbox.x1 - fig_bbox.x0) / (np.log10(data_bbox.x1) - np.log10(data_bbox.x0))
    xf = fig_bbox.x0 + kx*(np.log10(xd) - np.log10(data_bbox.x0))
    
    yd = t['detot']/t['etot']
    ky = (fig_bbox.y1 - fig_bbox.y0) / (np.log10(data_bbox.y1) - np.log10(data_bbox.y0))
    yf = fig_bbox.y0 + ky*(np.log10(yd) - np.log10(data_bbox.y0))
    
    aw = 0.06
    ah = 0.5*aw
    
    for i in range(Nplot):
        plt.axes([xf[i]-0.5*aw, yf[i]-0.5*ah, aw, ah], projection='mollweide')
        
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        
        plt.plot(cg.l.wrap_at(180*u.deg).radian[::100], cg.b.radian[::100], 'ko', ms=0.5, mew=0, alpha=0.2)
        plt.axis('off')
    
    plt.savefig('../plots/streaminess_dedl_{:d}.png'.format(Nplot))


# Streaminess

def stream_frames(hid=523889, lowmass=True):
    """Determine reference frames aligned with each stream progenitor's orbit"""
    
    orbits = pickle.load(open('../data/orbits_progenitors_halo.{:d}_lowmass.{:d}.pkl'.format(hid, lowmass), 'rb'))
    print(np.shape(orbits.x))
    N = np.shape(orbits.x)[1]
    
    frames = []
    #N = 10
    
    for i in range(N):
        orbit = orbits[:,i]
        c = orbit.to_coord_frame(coord.ICRS())
        
        # pick 10Myr point in the past, 10Myr point in the future
        endpoints = [c[0], c[10]]
        
        # frame from endpoints
        frame = gc.GreatCircleICRSFrame.from_endpoints(endpoints[0], endpoints[1], origin=c[0])
        cs = c.transform_to(frame)
        #print(cs[0])
        frames += [frame]
    
    # output
    pickle.dump(frames, open('../data/frames_halo.{:d}_lowmass.{:d}.pkl'.format(hid, lowmass), 'wb'))

def sky_dispersion(hid=523889, lowmass=True, f=0.2):
    """"""
    t = Table.read('../data/streams_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = np.isfinite(t['l']) #& (t['dl']/t['l']>1e-2)
    to_plot = ind_all[ind_done][:]
    
    frames = pickle.load(open('../data/frames_halo.{:d}_lowmass.{:d}.pkl'.format(hid, lowmass), 'rb'))
    
    print(N, len(frames))
    
    t['std_phi1'] = np.empty(N) * np.nan * u.deg
    t['std_phi2'] = np.empty(N) * np.nan * u.deg
    t['std_l'] = np.empty(N) * np.nan * u.deg
    t['std_b'] = np.empty(N) * np.nan * u.deg
    t['med_l'] = np.empty(N) * np.nan * u.deg
    t['med_b'] = np.empty(N) * np.nan * u.deg
    
    #N = 100
    N = np.sum(ind_done)
    #N = 10

    for i in range(N):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        
        cs = cg.transform_to(frames[to_plot[i]])
        Nstar = np.size(cs.phi1)
        Ncalc = int(f*Nstar)
        
        t['std_phi1'][to_plot[i]] = np.std(cs.phi1.deg[:Ncalc])
        t['std_phi2'][to_plot[i]] = np.std(cs.phi2.deg[:Ncalc])
        t['std_l'][to_plot[i]] = np.std(cg.l.wrap_at(180*u.deg).deg)
        t['std_b'][to_plot[i]] = np.std(cg.b.deg)
        t['med_l'][to_plot[i]] = np.median(cg.l.wrap_at(180*u.deg).deg)
        t['med_b'][to_plot[i]] = np.median(cg.b.deg)

    t.write('../data/streams_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass), overwrite=True)

    plt.close()
    plt.figure()
    
    plt.plot(cs.phi1.deg[:], cs.phi2.deg[:], 'ko', ms=2)
    plt.plot(cs.phi1.deg[:Ncalc], cs.phi2.deg[:Ncalc], 'ro', ms=2)
    print(np.std(cs.phi2.deg[:Ncalc]))
    #plt.scatter(t['std_phi1'], t['std_phi2'], c=t['std_b'], vmin=0, vmax=20, cmap='viridis_r')
    
    plt.gca().set_aspect('equal')
    plt.tight_layout()

def plot_spread(hid=523889, lowmass=True):
    """"""
    t = Table.read('../data/streams_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = np.isfinite(t['l']) #& (t['dl']/t['l']>1e-2)
    to_plot = ind_all[ind_done][:]
    
    phi = np.linspace(0,np.pi*0.5,100)
    r = 35
    x_ = r * np.cos(phi)
    y_ = r * np.sin(phi)
    
    plt.close()
    plt.figure(figsize=(12,8))
    
    plt.scatter(t['std_phi1'], t['std_phi2'], c=t['std_b'], vmax=20, s=5, cmap='viridis_r')
    #plt.scatter(t['std_phi1'], t['std_phi2'], c=t['circLz']*np.sign(t['lz']), vmax=-0.2, cmap='viridis_r', s=5)
    
    plt.plot(x_, y_, 'r-', lw=0.5)
    
    plt.gca().set_aspect('equal')
    plt.tight_layout()

def plot_morphology_sky(hid=523889, lowmass=True, halo=True, Nplot=100, log=False):
    """"""
    
    t = Table.read('../data/streams_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = np.isfinite(t['l']) #& (t['dl']/t['l']>1e-2)
    
    to_plot = ind_all[ind_done]
    np.random.seed(381)
    to_plot = np.random.choice(to_plot, size=Nplot, replace=False)
    
    t = t[to_plot]
    
    x_ = np.sqrt(t['med_l']**2 + t['med_b']**2)
    y_ = np.abs(t['lz']/t['l'])
    
    plt.close()
    plt.figure(figsize=(12,12))
    
    #plt.scatter(t['std_phi1'], t['std_phi2'], c=t['circLz']*np.sign(t['lz']), vmax=-0.2, cmap='viridis_r', s=2)
    #plt.scatter(x_, t['std_phi2'], c=t['circLz']*np.sign(t['lz']), vmax=-0.2, cmap='viridis_r', s=2)
    plt.scatter(x_, y_, c=t['circLz']*np.sign(t['lz']), vmax=-0.2, cmap='viridis_r', s=2)
    
    if log:
        plt.gca().set_xscale('log')
        #plt.gca().set_yscale('log')
    
    plt.xlim(0.02,200)
    plt.ylim(0.01,1)
    #plt.ylim(3,40)
    
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
        
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        
        plt.plot(cg.l.wrap_at(180*u.deg).radian[::50], cg.b.radian[::50], 'ko', ms=0.5, mew=0, alpha=0.2)
        plt.axis('off')
    
    plt.savefig('../plots/streaminess_sky_{:d}.png'.format(Nplot))

def sky_morphology(hid=523889, lowmass=True, halo=True):
    """"""
    
    t = Table.read('../data/streams_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = np.isfinite(t['l']) & (t['dl']/t['l']>1e-2)
    to_plot = ind_all[ind_done][:]
    
    t = t[to_plot]
    N = len(t)
    
    sigma_l = np.zeros(N) * u.deg
    sigma_b = np.zeros(N) * u.deg
    med_l = np.zeros(N) * u.deg
    med_b = np.zeros(N) * u.deg
    
    for i in range(N):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        
        med_l[i] = np.median(cg.l.wrap_at(180*u.deg))
        med_b[i] = np.median(cg.b)
        sigma_l[i] = np.std(cg.l.wrap_at(180*u.deg))
        sigma_b[i] = np.std(cg.b)
    
    med_sky = np.sqrt(med_b**2 + med_l**2)
    
    plt.close()
    plt.figure()
    
    #plt.plot(sigma_l, sigma_b, 'ko', mew=0, ms=4, alpha=0.3)
    #plt.scatter(sigma_l, sigma_b, c=med_sky.value, norm=mpl.colors.LogNorm(vmin=2, vmax=10))
    plt.scatter(sigma_l, sigma_b, c=med_sky.value, vmin=2, vmax=10, s=5)
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    plt.xlabel('$\sigma_l$ [deg]')
    plt.ylabel('$\sigma_b$ [deg]')
    
    plt.tight_layout()
    plt.savefig('../plots/sky_sigma_galactic.png')


# Stream properties

def get_rgal(hid=523889, lowmass=True, halo=True, test=False, graph=False):
    """Calculate 25,50,75 percentiles of each stream's galactocentric radial distribution, and store in the summary table
    test -- limit to first 10 streams
    graph -- output histograms of streams' radial distributions, with each stream a page in the output pdf
    """
    
    t = Table.read('../data/streams_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = np.isfinite(t['l'])
    
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    
    if test:
        Nplot = 10
    
    t['rgal_stream'] = np.zeros(N) * np.nan * u.kpc
    t['rgal_25'] = np.zeros(N) * np.nan * u.kpc
    t['rgal_75'] = np.zeros(N) * np.nan * u.kpc
    
    if graph:
        pp = PdfPages('../plots/rgal_streams.pdf')
    
    for i in range(Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        cgal = cg.transform_to(coord.Galactocentric())
        
        rgal = cgal.spherical.distance.to(u.kpc).value
        
        t['rgal_stream'][to_plot[i]] = np.median(rgal)
        t['rgal_25'][to_plot[i]] = np.percentile(rgal, 25)
        t['rgal_75'][to_plot[i]] = np.percentile(rgal, 75)
        
        if graph:
            plt.close()
            plt.figure(figsize=(10,6))
        
            plt.hist(rgal, bins=50, histtype='step', lw=2, label='Stream {:d}'.format(to_plot[i]))
            plt.axvline(np.median(rgal), lw=1.5, color='r', label='Median: {:.2f} kpc'.format(np.median(rgal)))
            plt.axvline(np.percentile(rgal, 25), lw=1.5, color='r', ls=':', label='IQR: {:.2f} kpc'.format(np.percentile(rgal, 75) - np.percentile(rgal, 25)))
            plt.axvline(np.percentile(rgal, 75), lw=1.5, color='r', ls=':', label='')
            
            plt.legend(fontsize='small')
            plt.xlabel('R$_{Gal}$ [kpc]')
            plt.ylabel('Number')
            
            plt.tight_layout()
            pp.savefig()
    
    if graph:
        pp.close()
    
    t.pprint()
    t.write('../data/streams_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass), overwrite=True)

def rgal_hist(hid=523889, lowmass=True, halo=True):
    """Plot median, 25% and 75% of the streams' galactocentric radii"""
    
    t = Table.read('../data/streams_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = np.isfinite(t['l'])
    
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    
    print('max rgal:', np.max(t['rgal_stream'][ind_done]))
    print('50, 90 rgal:', np.percentile(np.array(t['rgal_stream'][ind_done]), [50,90]))
    
    plt.close()
    plt.figure(figsize=(10,6))
    
    plt.hist(t['rgal_25'][ind_done], bins=np.logspace(-0.8,2.1,100), histtype='stepfilled', color=mpl.cm.Blues(0.5), alpha=0.2, label='25%')
    plt.hist(t['rgal_75'][ind_done], bins=np.logspace(-0.8,2.1,100), histtype='stepfilled', color=mpl.cm.Blues(0.9), alpha=0.2, label='75%')
    plt.hist(t['rgal_stream'][ind_done], bins=np.logspace(-0.8,2.1,100), histtype='step', color=mpl.cm.Blues(0.7), lw=2, label='Median')
    
    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')
    
    plt.legend(fontsize='small', loc=2)
    plt.xlabel('$R_{Gal}$ [kpc]')
    plt.ylabel('Number')
    
    plt.tight_layout()
    plt.savefig('../plots/rgal_total.png')


def sb_levels(hid=523889, lowmass=True, halo=True, test=False, i0=0):
    """Calculate stream surface brightness in healpix for a range of healpix levels"""
    
    t = Table.read('../data/streams_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = np.isfinite(t['l'])
    
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    
    if test:
        Nplot = i0 + 1
    else:
        i0 = 0
    
    # Prepare the healpix pixels
    m0 = -48.60
    #m0 = 0
    levels = np.arange(6,12.1,1,dtype=int)
    
    plt.close()
    plt.figure(figsize=(12,7))
    
    for i in range(i0, Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        
        #ind_lim = pkl['g']<glim
        #cg = cg[ind_lim]
        
        flux = 10**(-(pkl['g'] - m0)/2.5)
        
        for j, level in enumerate(levels):
            NSIDE = int(2**level)
            da = (hp.nside2pixarea(NSIDE, degrees=True)*u.deg**2).to(u.arcsec**2)
            
            ind_pix = hp.ang2pix(NSIDE, cg.l.deg, cg.b.deg, nest=False, lonlat=True)
            
            flux_tot = np.zeros(hp.nside2npix(NSIDE))
            
            for e, ind in enumerate(ind_pix):
                flux_tot[ind] += flux[e]
            
            ind_detected = flux_tot > 0
            s = -2.5*np.log10(flux_tot[ind_detected]/da.to(u.arcsec**2).value) + m0
            ind_finite = np.isfinite(s)
            
            area = np.sum(ind_detected) * da.to(u.arcsec**2)
            mu = -2.5*np.log10(np.sum(flux_tot[ind_detected])/area.to(u.arcsec**2).value) + m0
            print(mu)
            
            color = mpl.cm.Blues(0.1+j/(np.size(levels)))
            
            plt.hist(s[ind_finite], bins=50, log=True, density=False, histtype='step', color=color, label='level {:d}, pix={:.2f}, $\mu_{{tot}}={:.1f}$ mag arcsec$^{{-2}}$'.format(level, np.sqrt(da).to(u.deg), mu))
            plt.axvline(mu, color=color, lw=0.5, ls='--')
    
    plt.legend(fontsize='x-small', loc=2)
    plt.xlabel('$\mu$ [mag arsec$^{-2}$]')
    plt.ylabel('Number')
    
    plt.tight_layout()
    plt.savefig('../plots/sb_levels_halo.{:d}_stream.{:04d}.png'.format(hid, to_plot[i]))

def sb_glim(hid=523889, lowmass=True, halo=True, test=False, level=8, i0=0):
    """Calculate stream surface brightness in healpix for a range of limiting magnitudes"""
    
    t = Table.read('../data/streams_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = np.isfinite(t['l'])
    
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    print(Nplot)
    
    if test:
        Nplot = i0 + 1
    else:
        i0 = 0
    
    # Prepare the healpix pixels
    m0 = -48.60
    glims = np.arange(22,27.1,1)
    
    NSIDE = int(2**level)
    da = (hp.nside2pixarea(NSIDE, degrees=True)*u.deg**2).to(u.arcsec**2)
    
    
    plt.close()
    plt.figure()
    
    for i in range(i0,Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        
        for j, glim in enumerate(glims):
            ind_lim = pkl['g']<glim
            cg = pkl['cg'][ind_lim]
            
            flux = 10**(-(pkl['g'][ind_lim] - m0)/2.5)
            ind_pix = hp.ang2pix(NSIDE, cg.l.deg, cg.b.deg, nest=False, lonlat=True)
            
            flux_tot = np.zeros(hp.nside2npix(NSIDE))
            
            for e, ind in enumerate(ind_pix):
                flux_tot[ind] += flux[e]
            
            s = -2.5*np.log10(flux_tot/da.to(u.arcsec**2).value) + m0
            ind_finite = np.isfinite(s)
            
            area = np.sum(ind_finite) * da.to(u.arcsec**2)
            mu = -2.5*np.log10(np.sum(flux_tot[ind_finite])/area.to(u.arcsec**2).value) + m0
            
            color = mpl.cm.Blues(0.1+j/(np.size(glims)))
            
            plt.hist(s[ind_finite], bins=50, log=True, density=False, histtype='step', color=color, label='g$_{{lim}}$={:.0f}mag, $\mu_{{tot}}$={:.1f}mag arcsec$^{{-2}}$'.format(glim, mu))
            plt.axvline(mu, color=color, lw=0.5, ls='--')
    
    plt.legend(fontsize='x-small', loc=2)
    plt.xlabel('$\mu$ [mag arsec$^{-2}$]')
    plt.ylabel('Number')
    
    plt.tight_layout()
    plt.savefig('../plots/sb_glims_halo.{:d}_stream.{:04d}_level.{:d}.png'.format(hid, to_plot[i], level))

def demonstrate_sb(hid=523889, lowmass=True, i0=8980, level=8):
    """"""
    t = Table.read('../data/streams_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    
    pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, i0), 'rb'))
    print(pkl.keys())
    
    wangle = 180*u.deg
    
    m0 = -48.60
    glims = np.arange(22,27.1,1)
    
    NSIDE = int(2**level)
    da = (hp.nside2pixarea(NSIDE, degrees=True)*u.deg**2).to(u.arcsec**2)
    
    plt.close()
    fig, ax = plt.subplots(2,3,figsize=(12,6), sharex='col', sharey='col')
    
    for i, glim in enumerate([22,27]):
        ind_lim = pkl['g']<glim
        cg = pkl['cg'][ind_lim]
        print(np.sum(ind_lim), np.shape(cg.l))
        
        plt.sca(ax[i][0])
        plt.scatter(-cg.l.wrap_at(wangle), cg.b, s=5*pkl['g'][ind_lim]**-1, color='k', ec='none', alpha=0.2)
        
        plt.ylabel('g < {:.0f} mag\nb [deg]'.format(glim))
        plt.gca().set_aspect('equal')
        
        # surface brightness
        flux = 10**(-(pkl['g'][ind_lim] - m0)/2.5)
        ind_pix = hp.ang2pix(NSIDE, cg.l.deg, cg.b.deg, nest=False, lonlat=True)
        
        flux_tot = np.zeros(hp.nside2npix(NSIDE))
        
        for e, ind in enumerate(ind_pix):
            flux_tot[ind] += flux[e]
        
        s = -2.5*np.log10(flux_tot/da.to(u.arcsec**2).value) + m0
        ind_finite = np.isfinite(s)
        
        area = np.sum(ind_finite) * da.to(u.arcsec**2)
        mu = -2.5*np.log10(np.sum(flux_tot[ind_finite])/area.to(u.arcsec**2).value) + m0
        
        
        plt.sca(ax[i][1])
        plt.axis('off')
        hp.mollview(s, nest=False, fig=fig, sub=232+i*3, cmap='Blues_r', cbar=False, badcolor='w', title='', min=28, max=42)
        hp.visufunc.graticule(dpar=30, dmer=60)
        
        color = 'k'
        
        plt.sca(ax[i][2])
        plt.hist(s[ind_finite], bins=50, log=True, density=False, histtype='step', color=color, label='')
        plt.axvline(mu, color=color, lw=0.5, ls='--', label='$\mu_{{tot}}$ = {:.1f} mag arcsec$^{{-2}}$'.format(mu))
        plt.legend(fontsize='xx-small')
        plt.ylabel('Number')
        
        if i==0:
            plt.sca(ax[1][2])
            plt.hist(s[ind_finite], bins=50, log=True, density=False, histtype='step', color=color, label='', alpha=0.4)
    
    plt.sca(ax[1][0])
    plt.xlabel('l [deg]')
    
    plt.sca(ax[1][2])
    plt.xlabel('$\mu$ [mag arcsec$^{-2}$]')
    
    plt.tight_layout()
    plt.savefig('../plots/sb_demonstration_level.{:d}.png'.format(level))

def get_sb(hid=523889, lowmass=True, halo=True, test=False, graph=False):
    """"""
    t = Table.read('../data/streams_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    ind_all = np.arange(N, dtype=int)
    ind_done = np.isfinite(t['l'])
    
    to_plot = ind_all[ind_done]
    Nplot = np.size(to_plot)
    
    if test:
        Nplot = 10
    
    #t['mu_6_27.6'] = np.zeros(N) * np.nan * u.mag * u.arcsec**-2
    #t['mu_8_27.6'] = np.zeros(N) * np.nan * u.mag * u.arcsec**-2
    #t['mu_10_27.6'] = np.zeros(N) * np.nan * u.mag * u.arcsec**-2
    #t['mu_6_22.0'] = np.zeros(N) * np.nan * u.mag * u.arcsec**-2
    #t['mu_8_22.0'] = np.zeros(N) * np.nan * u.mag * u.arcsec**-2
    #t['mu_10_22.0'] = np.zeros(N) * np.nan * u.mag * u.arcsec**-2
    
    # surface brightness setup
    wangle = 180*u.deg
    m0 = -48.60
    
    glims = np.array([22, 27.6])
    levels = np.array([6,7,8,9,10], dtype=int)
    
    for level in levels:
        t['mu_{:d}'.format(level)] = np.zeros(N) * np.nan * u.mag * u.arcsec**-2
    
    #if graph:
        #pp = PdfPages('../plots/rgal_streams.pdf')
    
    for i in range(Nplot):
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, to_plot[i]), 'rb'))
        cg = pkl['cg']
        #cgal = cg.transform_to(coord.Galactocentric())
        
        # calculate surface brightness
        #for glim in glims:
            #ind_lim = pkl['g']<glim
            #cg = pkl['cg'][ind_lim]
            
        for level in levels:
            NSIDE = int(2**level)
            da = (hp.nside2pixarea(NSIDE, degrees=True)*u.deg**2).to(u.arcsec**2)
            
            #flux = 10**(-(pkl['g'][ind_lim] - m0)/2.5)
            flux = 10**(-(pkl['g'] - m0)/2.5)
            ind_pix = hp.ang2pix(NSIDE, cg.l.deg, cg.b.deg, nest=False, lonlat=True)
            
            flux_tot = np.zeros(hp.nside2npix(NSIDE))
            
            for e, ind in enumerate(ind_pix):
                flux_tot[ind] += flux[e]
            
            #s = -2.5*np.log10(flux_tot/da.to(u.arcsec**2).value) + m0
            #ind_finite = np.isfinite(s)
            ind_finite = flux_tot>0
            area = np.sum(ind_finite) * da.to(u.arcsec**2)
            mu = -2.5*np.log10(np.sum(flux_tot[ind_finite])/area.to(u.arcsec**2).value) + m0
            
            #t['mu_{:d}_{:.1f}'.format(level, glim)][to_plot[i]] = mu
            t['mu_{:d}'.format(level)][to_plot[i]] = mu
        
        #if graph:
            #plt.close()
            #plt.figure(figsize=(10,6))
        
            #plt.hist(rgal, bins=50, histtype='step', lw=2, label='Stream {:d}'.format(to_plot[i]))
            #plt.axvline(np.median(rgal), lw=1.5, color='r', label='Median: {:.2f} kpc'.format(np.median(rgal)))
            #plt.axvline(np.percentile(rgal, 25), lw=1.5, color='r', ls=':', label='IQR: {:.2f} kpc'.format(np.percentile(rgal, 75) - np.percentile(rgal, 25)))
            #plt.axvline(np.percentile(rgal, 75), lw=1.5, color='r', ls=':', label='')
            
            #plt.legend(fontsize='small')
            #plt.xlabel('R$_{Gal}$ [kpc]')
            #plt.ylabel('Number')
            
            #plt.tight_layout()
            #pp.savefig()
    
    #if graph:
        #pp.close()
    
    t.pprint()
    t.write('../data/streams_sb_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass), overwrite=True)



# visualization

def save_healpix(hid=523889, lowmass=True, glim=27, level=8, full=True):
    """Save total fluxes of stream stars in a healpix map"""
    
    t = Table.read('../data/stream_progenitors_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    
    ind_all = np.arange(N, dtype=int)
    ind_halo = get_halo(t, full=full)
    to_run = ind_all[ind_halo]
    
    # Prepare the healpix pixels
    NSIDE = int(2**level)

    flux_tot = np.zeros(hp.nside2npix(NSIDE))
    mask = np.zeros(hp.nside2npix(NSIDE), dtype=bool)
    nstar = np.zeros(hp.nside2npix(NSIDE), dtype=int)
    da = (hp.nside2pixarea(NSIDE, degrees=True)*u.deg**2).to(u.arcmin**2)
    m0 = -48.60
    
    i = to_run[0]
    for i in to_run[:]:
        try:
            pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, i), 'rb'))
            cg = pkl['cg']
            ind_lim = pkl['g']<glim
            
            flux = 10**(-(pkl['g'][ind_lim] - m0)/2.5)
            
            ind_pix = hp.ang2pix(NSIDE, cg.l.deg[ind_lim], cg.b.deg[ind_lim], nest=False, lonlat=True)
            
            for e, ind in enumerate(ind_pix):
                flux_tot[ind] += flux[e]
                mask[ind] = True
                nstar[ind] += 1
        except FileNotFoundError:
            pass
    
    flux_tot[~mask] = np.nan
    
    np.savez('../data/flux_full.{:d}_halo.{:d}.{:d}_l.{:d}_g.{:.1f}'.format(full, hid, lowmass, level, glim), flux=flux_tot, n=nstar)

def flux_difference():
    """Make sure deeper map has more stars and flux in every healpix"""
    f1 = np.load('../data/flux_halo.523889.1_l.8_g.22.0.npz')
    f2 = np.load('../data/flux_halo.523889.1_l.8_g.27.0.npz')
    
    df = f2['flux'] - f1['flux']
    dn = f2['n'] - f1['n']
    
    print(np.min(dn), np.max(dn))
    
    print(np.nanmax(df), np.nanmin(df))
    print(np.sum(df>0), np.sum(df<0))
    
    plt.close()
    plt.figure()
    
    plt.hist(df, bins=np.logspace(-40,-22))
    
    plt.gca().set_xscale('log')
    plt.tight_layout()

def plot_sb(hid=523889, lowmass=True, glim=27, level=8, full=True):
    """"""
    
    f = np.load('../data/flux_full.{:d}_halo.{:d}.{:d}_l.{:d}_g.{:.1f}.npz'.format(full, hid, lowmass, level, glim))
    
    # Prepare the healpix pixels
    NSIDE = int(2**level)

    #flux_tot = np.zeros(hp.nside2npix(NSIDE))
    da = (hp.nside2pixarea(NSIDE, degrees=True)*u.deg**2).to(u.arcmin**2)
    m0 = -48.60
    
    s = -2.5*np.log10(f['flux']/da.to(u.arcsec**2).value) + m0

    # Plot the pixelization
    cmap = 'Blues'
    vmin = 25
    vmax = 35
    
    plt.close()
    fig = plt.figure(figsize=(10,5))
    
    #hp.mollview(s, nest=False, fig=fig, cmap=cmap, min=vmin, max=vmax, norm='log', cbar=False)
    hp.mollview(s, nest=False, fig=fig, cmap=cmap, min=vmin, max=vmax, norm=None, cbar=False, badcolor='k')
    
    plt.grid()
    plt.title('g<{:g}'.format(glim), fontsize='medium')
    
    im = plt.gca().get_images()[0]
    #plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap), ax=plt.gca(), format=mpl.ticker.ScalarFormatter(), label='Surface brightness [mag arcsec$^2$]')
    plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap), ax=plt.gca(), format=mpl.ticker.ScalarFormatter(), label='Surface brightness [mag arcsec$^2$]')
    #plt.gca().invert_yaxis()
    
    plt.tight_layout()
    #plt.savefig('../plots/allstreams_sb_g.{:.1f}_norm.log.png'.format(glim))
    plt.savefig('../plots/allstreams_sb_full.{:d}_g.{:.1f}_norm.lin_level.{:d}.png'.format(full, glim, level), dpi=400)


def sb_single(hid=523889, lowmass=True, glim=27, level=8):
    """"""
    
    t = Table.read('../data/stream_progenitors_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    
    ind_halo = get_halo(t)
    ind_all = np.arange(N, dtype=int)
    to_run = ind_all[ind_halo]
    
    # Prepare the healpix pixels
    NSIDE = int(2**level)

    flux_tot = np.zeros(hp.nside2npix(NSIDE))
    da = (hp.nside2pixarea(NSIDE, degrees=True)*u.deg**2).to(u.arcmin**2)
    m0 = -48.60
    
    #print(np.sqrt(da), da)
    
    i = to_run[0]
    for i in to_run[:]:
        try:
            pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, i), 'rb'))
            cg = pkl['cg']
            ind_lim = pkl['g']<glim
            cg = cg[ind_lim]
            
            flux = 10**(-(pkl['g'] - m0)/2.5)
            
            ind_pix = hp.ang2pix(NSIDE, cg.l.deg, cg.b.deg, nest=False, lonlat=True)
            
            for e, ind in enumerate(ind_pix):
                flux_tot[ind] += flux[e]
        except FileNotFoundError:
            pass
    
    ind_nan = flux_tot==0
    flux_tot[ind_nan] = np.nan
    #print(np.min(flux_tot[~ind_nan]))
    
    s = -2.5*np.log10(flux_tot/da.to(u.arcsec**2).value) + m0
    print(s[:10])
    

    # Plot the pixelization
    cmap = 'Blues_r'
    vmin = 32
    vmax = 40
    
    plt.close()
    fig = plt.figure(figsize=(10,5))
    
    #hp.mollview(s, nest=False, fig=fig, cmap=cmap, min=vmin, max=vmax, norm='log', cbar=False)
    hp.mollview(s, nest=False, fig=fig, cmap=cmap, min=vmin, max=vmax, norm=None, cbar=False, badcolor='w')
    
    plt.grid()
    plt.title('g<{:g}'.format(glim), fontsize='medium')
    
    im = plt.gca().get_images()[0]
    #plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap), ax=plt.gca(), format=mpl.ticker.ScalarFormatter(), label='Surface brightness [mag arcsec$^2$]')
    plt.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap), ax=plt.gca(), format=mpl.ticker.ScalarFormatter(), label='Surface brightness [mag arcsec$^2$]')
    
    plt.tight_layout()
    #plt.savefig('../plots/allstreams_sb_g.{:.1f}_norm.log.png'.format(glim))
    plt.savefig('../plots/allstreams_sb_g.{:.1f}_norm.lin_level.{:d}.png'.format(glim, level))

def plot_stream(hid=523889, lowmass=True, glim=27, level=8, plot_map=True):
    """Save total fluxes of stream stars in a healpix map"""
    
    t = Table.read('../data/stream_progenitors_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    
    ind_halo = get_halo(t)
    ind_all = np.arange(N, dtype=int)
    to_run = ind_all[ind_halo]
    
    # Prepare the healpix pixels
    NSIDE = int(2**level)

    flux_tot = np.zeros(hp.nside2npix(NSIDE))
    mask = np.zeros(hp.nside2npix(NSIDE), dtype=bool)
    nstar = np.zeros(hp.nside2npix(NSIDE), dtype=int)
    da = (hp.nside2pixarea(NSIDE, degrees=True)*u.deg**2).to(u.arcmin**2)
    m0 = -48.60
    
    i = to_run[1]
    #for i in to_run[:]:
    try:
        pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, i), 'rb'))
        cg = pkl['cg']
        ind_lim = pkl['g']<glim
        #cg = cg[ind_lim]
        print(np.sum(ind_lim))
        print(np.sum(pkl['mass']))
        print(10**t['logMgc_at_birth'][i])
        
        # read isochrone
        age = np.around(t['t_form'][i], decimals=1)
        feh = np.around(t['FeH'][i], decimals=1)
        iso = read_isochrone(age=age*u.Gyr, feh=feh, ret=True)
        
        # interpolate isochrone
        bbox = [np.min(iso['initial_mass']), np.max(iso['initial_mass'])]

        # sample masses
        masses = sample_kroupa(t['logMgc_at_birth'][i])
        print(np.sum(masses))
        ind_alive = masses<bbox[1]
        masses = masses[ind_alive]
        print(np.sum(masses))
        
        
        flux = 10**(-(pkl['g'][ind_lim] - m0)/2.5)
        
        ind_pix = hp.ang2pix(NSIDE, cg.l.deg[ind_lim], cg.b.deg[ind_lim], nest=False, lonlat=True)
        
        for e, ind in enumerate(ind_pix):
            flux_tot[ind] += flux[e]
            mask[ind] = True
            nstar[ind] += 1
    except FileNotFoundError:
        pass
    
    flux_tot[~mask] = np.nan
    s = -2.5*np.log10(flux_tot/da.to(u.arcsec**2).value) + m0
    
    # Plot the pixelization
    cmap = 'Blues'
    vmin = 32
    vmax = 40
    
    wangle = 180*u.deg
    
    plt.close()
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111, projection='mollweide')
    
    if plot_map:
        hp.mollview(s, nest=False, fig=fig, cmap=cmap, norm=None, cbar=False, badcolor='k')
        plt.plot(-cg.l.wrap_at(wangle).rad[ind_lim], cg.b.rad[ind_lim], 'ro', mew=0, ms=2, alpha=0.1)
    else:
        plt.plot(-cg.l.wrap_at(wangle).rad[ind_lim], cg.b.rad[ind_lim], 'ro', mew=0, ms=2, alpha=0.1)
    
    plt.tight_layout()
    
    
    #plt.grid()
    plt.title('')
    plt.savefig('../plots/diag_stream.{:d}_map.{:d}.png'.format(i, plot_map))
    #plt.title('g<{:g}'.format(glim), fontsize='medium')
    
    #np.savez('../data/flux_halo.{:d}.{:d}_l.{:d}_g.{:.1f}'.format(hid, lowmass, level, glim), flux=flux_tot, n=nstar)


def plot_distance(hid=523889, lowmass=True, glim=27):
    """"""
    
    t = Table.read('../data/stream_progenitors_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    
    ind_halo = get_halo(t, full=False)
    ind_all = np.arange(N, dtype=int)
    to_run = ind_all[ind_halo]
    print(np.size(to_run))
    
    plt.close()
    fig = plt.figure(figsize=(12,6), facecolor='k')
    ax = fig.add_subplot(111, projection='mollweide')
    
    wangle = 180*u.deg
    dmin = 1*u.kpc
    dmax = 50*u.kpc
    
    for i in to_run[:]:
        try:
            pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, i), 'rb'))
            cg = pkl['cg']
            ind_lim = pkl['g']<glim
            
            plt.scatter(cg.l.wrap_at(wangle).rad[ind_lim], cg.b.rad[ind_lim], c=cg.distance.value[ind_lim], cmap='plasma', vmin=dmin.value, vmax=dmax.value, s=0.1, alpha=0.1)
            
        except FileNotFoundError:
            pass
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../plots/halo_streams_distance_g.{:.1f}.png'.format(glim), dpi=250, facecolor='k')

def plot_progenitors(hid=523889, lowmass=True):
    """"""
    
    # globular clusters
    t = Table.read('../data/mw_like_6.0_0.4_2.5_linear_survive.txt', format='ascii.commented_header', delimiter=' ')
    ind = t['haloID']==hid
    t = t[ind]
    
    c = coord.Galactocentric(x=t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc)
    cg = c.transform_to(coord.Galactic())
    
    # streams
    ts = Table.read('../data/mw_like_6.0_0.4_2.5_linear_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    ind_ = ts['haloID']==hid
    ts = ts[ind_]
    
    cs = coord.Galactocentric(x=ts['x']*u.kpc, y=ts['y']*u.kpc, z=ts['z']*u.kpc)
    cg_s = cs.transform_to(coord.Galactic())
    
    
    wangle = 180*u.deg
    dmin = 1*u.kpc
    dmax = 50*u.kpc
    
    plt.close()
    fig = plt.figure(figsize=(13,6))
    fig.add_subplot(111, projection='mollweide')
    
    im = plt.scatter(cg.l.wrap_at(wangle).rad, cg.b.rad, c=cg.distance.value, vmin=dmin.value, vmax=dmax.value, s=5*(0.2*t['logMgc_at_birth'])**9, cmap='plasma', zorder=0, linewidth=0)
    
    #im_s = plt.scatter(cg_s.l.wrap_at(wangle).rad, cg_s.b.rad, c='w', s=10*(0.2*ts['logMgc_at_birth'])**9, zorder=1, linewidth=0)
    #im_s = plt.scatter(cg_s.l.wrap_at(wangle).rad, cg_s.b.rad, c=cg_s.distance.value, vmin=dmin.value, vmax=dmax.value, s=5*(0.2*ts['logMgc_at_birth'])**9, cmap='plasma', alpha=0.5, zorder=2, linewidth=0)
    
    plt.colorbar(im, label='Distance [kpc]')
    
    plt.axis('off')
    
    plt.tight_layout()
    
    
def save_healpix_distbins(hid=523889, lowmass=True, glim=27, level=8):
    """Save total fluxes of stream stars in distance bins as healpix maps"""
    
    t = Table.read('../data/stream_progenitors_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    
    ind_halo = get_halo(t, full=True)
    ind_all = np.arange(N, dtype=int)
    to_run = ind_all[ind_halo]
    
    # set up distance bins
    Nbin = 12
    #dbins = np.array([0,5,10,1000])*u.kpc
    
    dbins = np.linspace(1,15,Nbin-1)
    # set lower and upper limits
    dbins = np.insert(dbins, 0, 0)
    dbins = np.insert(dbins, Nbin, 1000) * u.kpc
    
    print(dbins)
    
    # Prepare the healpix pixels
    NSIDE = int(2**level)

    flux_tot = np.zeros((hp.nside2npix(NSIDE), Nbin))
    mask = np.zeros((hp.nside2npix(NSIDE), Nbin), dtype=bool)
    nstar = np.zeros((hp.nside2npix(NSIDE), Nbin), dtype=int)
    da = (hp.nside2pixarea(NSIDE, degrees=True)*u.deg**2).to(u.arcmin**2)
    m0 = -48.60
    
    i = to_run[0]
    for i in to_run[:]:
        try:
            pkl = pickle.load(open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, i), 'rb'))
            cg = pkl['cg']
            ind_lim = pkl['g']<glim
            
            for j in range(Nbin):
                ind_dist = (cg.distance>dbins[j]) & (cg.distance<=dbins[j+1])
                ind_slice = ind_dist & ind_lim
                
                flux = 10**(-(pkl['g'][ind_slice] - m0)/2.5)
                
                ind_pix = hp.ang2pix(NSIDE, cg.l.deg[ind_slice], cg.b.deg[ind_slice], nest=False, lonlat=True)
                
                for e, ind in enumerate(ind_pix):
                    flux_tot[ind][j] += flux[e]
                    mask[ind][j] = True
                    nstar[ind][j] += 1
        except FileNotFoundError:
            pass
    
    flux_tot[~mask] = np.nan
    
    np.savez('../data/flux_dist_halo.{:d}.{:d}_l.{:d}_g.{:.1f}'.format(hid, lowmass, level, glim), flux=flux_tot, n=nstar, dbins=dbins)

def plot_sb_distbins(hid=523889, lowmass=True, glim=27, level=8, individual=False):
    """"""
    
    f = np.load('../data/flux_dist_halo.{:d}.{:d}_l.{:d}_g.{:.1f}.npz'.format(hid, lowmass, level, glim))
    dbins = f['dbins']
    Nbin = len(dbins)-1
    
    # Prepare the healpix pixels
    NSIDE = int(2**level)

    #flux_tot = np.zeros(hp.nside2npix(NSIDE))
    da = (hp.nside2pixarea(NSIDE, degrees=True)*u.deg**2).to(u.arcmin**2)
    m0 = -48.60
    
    s = -2.5*np.log10(f['flux']/da.to(u.arcsec**2).value) + m0

    # Plot the pixelization
    cmap = 'Greys'
    vmin = 25
    vmax = 35
    
    if individual:
        for i in range(Nbin):
            plt.close()
            fig = plt.figure(facecolor='k')
            #vmin = 20
            #vmax = 30
            hp.mollview(s[:,i], nest=False, fig=fig, cmap=cmap, min=vmin, max=vmax, norm=None, cbar=False, badcolor='k', bgcolor='k', hold=True, title='', xsize=6000)
            plt.savefig('../plots/allstreams_sb_dbin.{:d}_g.{:.1f}_norm.lin_level.{:d}.png'.format(i, glim, level), dpi=600, facecolor=fig.get_facecolor())
    else:
        nrow = np.int(np.sqrt(Nbin))
        ncol = Nbin//nrow
        if Nbin>nrow*ncol:
            nrow +=1
        print(nrow, ncol)
        da = 4
        w = ncol * da
        h = nrow * da * 0.5
        
        plt.close()
        fig, ax = plt.subplots(nrow, ncol, figsize=(w,h), squeeze=False)
        
        for i in range(Nbin):
            irow = i//ncol
            icol = i%ncol
            plt.sca(ax[irow][icol])
            hp.mollview(s[:,i], nest=False, fig=fig, cmap=cmap, min=vmin, max=vmax, norm=None, cbar=False, badcolor='k', hold=True, title='')
            plt.title('{:.0f} < d < {:.0f}'.format(dbins[i], dbins[i+1]), fontsize='small')
    
def test_tint(i=4):
    """"""
    Nbin = 12
    image = io.imread('../plots/allstreams_sb_dbin.{:d}_g.27.0_norm.lin_level.8.png'.format(i))
    
    hsv = color.rgb2hsv(image[:,:,:3])
    
    tint_color = mpl.cm.magma((i+1)/Nbin)
    tint_rgb = mpl.colors.to_rgb(tint_color)
    tint_hsv = mpl.colors.rgb_to_hsv(tint_rgb)
    
    hsv[:, :, 0] = tint_hsv[0]
    hsv[:, :, 1] = tint_hsv[1]
    #hsv[:, :, 1] = 0.3
    
    #red_multiplier = np.array([1, 0, 0, 1])
    #yellow_multiplier = np.array([1, 1, 0, 1])

    plt.close()
    fig, ax = plt.subplots(ncols=2, figsize=(12, 3), sharex=True, sharey=True)
    
    plt.sca(ax[0])
    plt.imshow(image)
    plt.axis('off')
    
    plt.sca(ax[1])
    plt.imshow(color.hsv2rgb(hsv))
    plt.axis('off')
    
    plt.show()
    plt.tight_layout()
    
def combine(hid=523889, lowmass=True, glim=27, level=8):
    """"""
    f = np.load('../data/flux_dist_halo.{:d}.{:d}_l.{:d}_g.{:.1f}.npz'.format(hid, lowmass, level, glim))
    dbins = f['dbins']
    Nbin = len(dbins)-1
    
    nrow = np.int(np.sqrt(Nbin))
    ncol = Nbin//nrow
    da = 4
    w = ncol * da
    h = nrow * da * 0.5
    
    plt.close()
    fig, ax = plt.subplots(nrow, ncol, figsize=(w,h), sharex=True, sharey=True)
    
    for i in range(Nbin):
        image = io.imread('../plots/allstreams_sb_dbin.{:d}_g.{:.1f}_norm.lin_level.{:d}.png'.format(i, glim, level))
        
        hsv = color.rgb2hsv(image[:,:,:3])
        
        tint_color = mpl.cm.magma((i+1)/Nbin)
        tint_rgb = mpl.colors.to_rgb(tint_color)
        tint_hsv = mpl.colors.rgb_to_hsv(tint_rgb)
        
        hsv[:, :, 0] = tint_hsv[0]
        hsv[:, :, 1] = tint_hsv[1]
        
        image_tinted = color.hsv2rgb(hsv)
        
        irow = i//ncol
        icol = i%ncol
        plt.sca(ax[irow][icol])
        plt.imshow(image_tinted)
        plt.axis('off')
        
    plt.tight_layout()
    

def tint(hid=523889, lowmass=True, glim=27, level=8):
    """"""
    f = np.load('../data/flux_dist_halo.{:d}.{:d}_l.{:d}_g.{:.1f}.npz'.format(hid, lowmass, level, glim))
    dbins = f['dbins']
    Nbin = len(dbins)-1
    
    for i in range(Nbin):
        plt.close()
        fig = plt.figure(facecolor='k')
        
        image = io.imread('../plots/allstreams_sb_dbin.{:d}_g.{:.1f}_norm.lin_level.{:d}.png'.format(i, glim, level))
        
        hsv = color.rgb2hsv(image[:,:,:3])
        
        tint_color = mpl.cm.magma((i+1)/Nbin)
        tint_rgb = mpl.colors.to_rgb(tint_color)
        tint_hsv = mpl.colors.rgb_to_hsv(tint_rgb)
        
        hsv[:, :, 0] = tint_hsv[0]
        hsv[:, :, 1] = tint_hsv[1]
        
        image_tinted = color.hsv2rgb(hsv)
        
        #irow = i//ncol
        #icol = i%ncol
        #plt.sca(ax[irow][icol])
        plt.imshow(image_tinted)
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig('../plots/allstreams_sb_tint_dbin.{:d}_g.{:.1f}_norm.lin_level.{:d}.png'.format(i, glim, level), dpi=600, facecolor=fig.get_facecolor())

def blend(hid=523889, lowmass=True, glim=27, level=8):
    """"""
    f = np.load('../data/flux_dist_halo.{:d}.{:d}_l.{:d}_g.{:.1f}.npz'.format(hid, lowmass, level, glim))
    dbins = f['dbins']
    Nbin = len(dbins)-1
    
    img = []
    for i in range(Nbin):
        img += [io.imread('../plots/allstreams_sb_tint_dbin.{:d}_g.{:.1f}_norm.lin_level.{:d}.png'.format(i, glim, level))]
    
    img_blend = img[0]*0.
    alpha = 0.0002
    
    for i in range(Nbin):
        img_blend += alpha*img[i]*(i**1.5+0.2)
    
    plt.close()
    fig = plt.figure(facecolor='k')
    plt.imshow(img_blend)
    
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('../plots/allstreams_sb_blend_g.{:.1f}_norm.lin_level.{:d}.png'.format(glim, level), dpi=600, facecolor=fig.get_facecolor())
