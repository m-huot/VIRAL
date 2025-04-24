#!/bin/sh
#SBATCH -p shakhnovich,shared   # Partition to submit to
#SBATCH -N 1     # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 4     # cores requested
#SBATCH --mem=20000  # memory in Mb
#SBATCH -o bench_outfile  # send stdout to outfile (with job ID)
#SBATCH -e bench_errfile  # send stderr to errfile (with job ID)
#SBATCH -t 47:00:00  # time requested in hour:minute:second
export LD_LIBRARY_PATH=/n/home00/mhuot/.conda/envs/lantern/lib:$LD_LIBRARY_PATH
source activate lantern
cd ../plots

# Run the Python script with the specified embedding
python3 all_benchmark.py 
