from mock import *
import sys

if __name__ == '__main__':
    """Wrapper to create mock stellar streams"""
    
    args = sys.argv
    if len(args)<7:
        args += [False]
    
    mock_stream(hid=int(args[1]), graph=False, i0=0, f=args[2], lowmass=True, verbose=False, halo=False, nskip=int(args[3]), istart=int(args[4]), remaining=int(args[5]), test=bool(args[6]))
