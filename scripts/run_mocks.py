from mock import *
import sys

if __name__ == '__main__':
    """Wrapper to create mock stellar streams"""
    
    args = sys.argv
    if len(args)<6:
        args += [False]
    
    mock_stream(hid=args[1], graph=False, i0=0, f=1, lowmass=True, verbose=False, halo=False, nskip=int(args[2]), istart=int(args[3]), remaining=int(args[4]), test=bool(args[5]))
