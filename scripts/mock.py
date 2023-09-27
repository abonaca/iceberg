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
from skimage import io, color

ham_mw = gp.Hamiltonian(gp.MilkyWayPotential())

bulgePot = ham_mw.potential['bulge']
diskPot = ham_mw.potential['disk']
haloPot = ham_mw.potential['halo']
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

def all_orbits(hid=0, lowmass=True, test=False):
    """Calculate orbits of all disrupted globular clusters in a halo"""
    
    # read table
    if lowmass:
        t = Table.read('../data/mw_like_6.0_0.4_2.5_linear_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    else:
        t = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    
    hid = np.unique(t['haloID'])[hid]
    ind = (t['haloID']==hid) & ((t['t_accrete']==-1) | (t['t_disrupt']<t['t_accrete']))
    t = t[ind]
    
    if test:
        t = t[:10]
    
    N = len(t)
    
    # setup coordinates
    c = coord.Galactocentric(x=-1*t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=-1*t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # setup potential
    bulgePot = ham_mw.potential['bulge']
    diskPot = ham_mw.potential['disk']
    haloPot = ham_mw.potential['halo']
    
    totalPot = gp.CCompositePotential(component1=bulgePot, component2=diskPot, component3=haloPot)
    ham = gp.Hamiltonian(totalPot)
    
    # integration setup
    dt = -1*u.Myr
    T = 5*u.Gyr
    nstep = np.abs(int((T/dt).decompose()))
    
    # integrate orbits
    orbit = ham.integrate_orbit(w0, dt=dt, n_steps=nstep)
    pickle.dump(orbit, open('../data/orbits_halo.{:d}.pkl'.format(hid), 'wb'))
    
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
    t.write('../data/stream_progenitors_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass), overwrite=True)
    
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

def circularity(hid=523889, lowmass=True):
    """"""
    
    t = Table.read('../data/stream_progenitors_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    
    # calculate circularity
    maxLcirc_vec = np.vectorize(maxLcirc)
    maxLcirc_arr = maxLcirc_vec(np.linspace(-0.265, 0, 1000))
    
    t['lmax'] = (np.interp(t['etot'], np.linspace(-0.265, 0, 1000), maxLcirc_arr))
    t['circLz'] = np.abs(t['lz']/t['lmax'])
    
    t.pprint()
    t.write('../data/stream_progenitors_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass), overwrite=True)


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


def mock_stream(hid=523889, test=True, graph=True, i0=0, f=0.3, lowmass=True, verbose=False, halo=True, istart=0, nskip=1):
    """Create a mock stream from a disrupted globular cluster"""
    
    #if lowmass:
        #t = Table.read('../data/mw_like_6.0_0.4_2.5_linear_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    #else:
        #t = Table.read('../data/mw_like_4.0_0.5_lambda_disrupt.txt', format='ascii.commented_header', delimiter=' ')
    
    #hid = np.unique(t['haloID'])[hid]
    #ind = (t['haloID'] == hid) & ((t['t_accrete']==-1) | (t['t_disrupt']<t['t_accrete']))
    #t = t[ind]
    t = Table.read('../data/stream_progenitors_halo.{:d}_lowmass.{:d}.fits'.format(hid, lowmass))
    N = len(t)
    
    # starting time of dissolution (birth for in-situ, accretion time for accreted)
    # caveat: missing fluff of stars potentially dissolved before accretion time
    ind_insitu = t['t_accrete']==-1
    t_start = t['t_accrete'][:]
    t_start[ind_insitu] = t['t_form'][ind_insitu][:]
    
    t_end = t['t_disrupt'][:]
    
    # present-day cluster positions
    c = coord.Galactocentric(x=-1*t['x']*u.kpc, y=t['y']*u.kpc, z=t['z']*u.kpc, v_x=-1*t['vx']*u.km/u.s, v_y=t['vy']*u.km/u.s, v_z=t['vz']*u.km/u.s)
    w0 = gd.PhaseSpacePosition(c.transform_to(gc_frame).cartesian)
    
    # define gravitational potential -- just MilkyWayPotential without the central nucleus
    # don't have the agama decompositions for the 1e4Msun halos
    #diskPot = gp.CylSplinePotential.from_file('../data/pot_disk_{:d}.pot'.format(hid), units=galactic)
    #diskPot = gp.CylSplinePotential.from_file('../data/pot_disk_450916.pot', units=galactic)
    
    #bulgePot = ham_mw.potential['bulge']
    #diskPot = ham_mw.potential['disk']
    #haloPot = ham_mw.potential['halo']
    
    # TNG best-fit potential
    bulgePot = gp.HernquistPotential(10**10.26*u.Msun, 1.05*u.kpc, units=galactic)
    diskPot = gp.MiyamotoNagaiPotential(10**10.70*u.Msun, 4.20*u.kpc, 0.006*u.kpc, units=galactic)
    haloPot = gp.NFWPotential(10**11.57*u.Msun, 11.59*u.kpc, c=0.943, units=galactic)
    
    totalPot = gp.CCompositePotential(component1=bulgePot, component2=diskPot, component3=haloPot)
    #totalPot = ham_mw.potential
    
    # integration set up
    ham = gp.Hamiltonian(totalPot)
    dt = -1*u.Myr
    
    # setup how stars get released from the progenitor
    state = np.random.RandomState(seed=291)
    df = ms.FardalStreamDF(random_state=state)

    prog_mass = 10**t['logMgc_at_birth']*u.Msun
    gen = ms.MockStreamGenerator(df, ham)
    
    if halo:
        ind_all = np.arange(N, dtype=int)
        ind_halo = get_halo(t)
        to_run = ind_all[ind_halo]
    else:
        to_run = np.arange(i0, N, 1, dtype=int)
    
    if test:
        to_run = [to_run[0],]
    
    for i in to_run[istart::nskip]:
        if verbose: print('gc {:d}, logM = {:.2f}'.format(i, t['logMgc_at_birth'][i]))
        
        # define number of steps to start of dissolution
        n_steps = int((t_start[i]*u.Gyr/np.abs(dt)).decompose())
        n_disrupted = int((t_end[i]*u.Gyr/np.abs(dt)).decompose())
        
        nsingle = int(np.abs((t_start[i]-t_end[i])/dt.to(u.Gyr).value))
        
        # read isochrone
        age = np.around(t['t_form'][i], decimals=1)
        feh = np.around(t['FeH'][i], decimals=1)
        iso = read_isochrone(age=age*u.Gyr, feh=feh, ret=True)
        
        # interpolate isochrone
        bbox = [np.min(iso['initial_mass']), np.max(iso['initial_mass'])]

        # sample masses
        masses = sample_kroupa(t['logMgc_at_birth'][i])
        ind_alive = masses<bbox[1]
        masses = masses[ind_alive]
        
        # subsample mass
        if f<1:
            nchoose = int(f * np.size(masses))
            masses = np.random.choice(masses, size=nchoose, replace=False)
        
        # sort masses (assuming lowest mass get kicked out first)
        masses = np.sort(masses)
        
        # for accreted clusters, account for stars tidally stripped in the original host galaxy
        if ~ind_insitu[i]:
            # length of time spent in the progenitor galaxy
            t_prog = t['t_form'][i] - t['t_accrete'][i]
            # f_dyn = m_acc / m_init
            f_dyn = 1 - 0.06*(t_prog)
            #print(f_dyn)
            #m_lost = m_init - m_acc
            
            mass_cumsum = np.cumsum(masses)
            ind_mass = np.argmin(np.abs(mass_cumsum - (1-f_dyn)*prog_mass[i].value))
            masses = masses[ind_mass:]
        
        # make it an even number of stars
        if np.size(masses)%2:
            masses = masses[:-1]
        
        nstar = np.size(masses)
        ntail = int(nstar/2)
        
        # uniformly release the exact number of stars
        navg = ntail//n_disrupted
        nrelease = np.ones(n_disrupted, dtype=int) * navg
        nextra = int(ntail - np.sum(nrelease))
        nrelease[:nextra] += 1
        
        # add extra steps
        nrelease = np.concatenate([nrelease, np.zeros(n_steps - n_disrupted + 1, dtype=int)])
        if verbose: print('Ntail: {:d}, Nrelease: {:d}'.format(ntail, np.sum(nrelease)))

        # create mock stream
        t1 = time.time()
        stream_, prog = gen.run(w0[i], prog_mass[i], dt=dt, n_steps=n_steps, n_particles=nrelease)
        t2 = time.time()
        if verbose: print('Runtime: {:g}'.format(t2-t1))
        
        ##################
        # paint photometry
        # get distance modulus
        cg = stream_.to_coord_frame(coord.Galactic())
        dm = 5*np.log10((cg.distance.to(u.pc)).value) - 5
        
        # interpolate isochrone
        order = 1
        interp_g = InterpolatedUnivariateSpline(iso['initial_mass'], iso['LSST_g'], k=order, bbox=bbox)
        interp_r = InterpolatedUnivariateSpline(iso['initial_mass'], iso['LSST_r'], k=order, bbox=bbox)
        
        # photometry (no uncertainties)
        r = interp_r(masses) + dm
        g = interp_g(masses) + dm
        
        #############
        # save stream
        outdict = dict(cg=cg, mass=masses, g=g, r=r)
        pickle.dump(outdict, open('../data/streams/halo.{:d}_stream.{:04d}.pkl'.format(hid, i), 'wb'))
        
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
            plt.savefig('../plots/streams/halo.{:d}_stream.{:04d}.png'.format(hid, i))


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

def needed_isochrones(hid=0):
    """Print ages and metallicities needed"""
    
    t = Table.read('../data/mw_like_6.0_0.4_2.5_linear_disrupt.txt', format='ascii.commented_header', delimiter=' ')
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

def read_isochrone(age=10.2*u.Gyr, feh=-2.5, graph=False, verbose=False, ret=True):
    """Return isochrone of a given age and metallicity (extracted from a joint file of a given metallicity)
    graph=True - plots interpolation into all LSST bands"""
    
    # read isochrone
    iso_full = Table.read('../data/isochrones/lsst/mist_gc_{:.1f}.cmd'.format(feh), format='ascii.commented_header', header_start=12)
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

def plot_sb(hid=523889, lowmass=True, glim=27, level=8, full=False):
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
