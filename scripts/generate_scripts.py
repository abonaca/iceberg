
def mock(hid=523889, Nskip=30, remaining=False, test=False):
    """Bar disk scripts"""
    
    if test:
        Ntot = 1
    else:
        Ntot = Nskip
    
    for i in range(Ntot):
        job_name = 'stream_{:d}.{:d}'.format(Nskip, i)
        sbatch_text = """#!/bin/bash

# Submit this script with: sbatch <this-filename>

#SBATCH --time=24:00:00   # walltime
#SBATCH --ntasks=1   # number of processor cores (i.e. tasks)
#SBATCH --nodes=1   # number of nodes
#SBATCH --mem-per-cpu=4G   # memory per CPU core
#SBATCH -J "{:s}"   # job name
#SBATCH --mail-user=abonaca@carnegiescience.edu   # email address

# Notify at the beginning, end of job and on failure.
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL


## /SBATCH -p general # partition (queue)
## /SBATCH -o slurm/slurm.%N.%j.out # STDOUT
## /SBATCH -e slurm/slurm.%N.%j.err # STDERR

srun /home/abonaca/local/bin/python run_mocks.py {:d} {:d} {:d} {:d}
""".format(job_name, hid, Nskip, i, remaining)
        
        print(sbatch_text)
        print(sbatch_text, file=open('slurm/sbatch_{:s}'.format(job_name), 'w'))
