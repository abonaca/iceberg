from mock import *
import sys

if __name__ == '__main__':
    """Wrapper to create mock stellar streams"""
    
    args = sys.argv
    if len(args)<4:
        args += [False]
    
    mock_stream(hid=523889, graph=False, i0=0, f=0.3, lowmass=True, verbose=False, halo=True, nskip=int(args[1]), istart=int(args[2]), test=bool(args[3]))
    #evolve_bar_stars(mw_label=args[1], Nskip=int(args[2]), iskip=int(args[3]), test=bool(args[4]), m=1e10*u.Msun, Nrand=4000)
