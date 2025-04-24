#!/bin/sh
#SBATCH -p shakhnovich,shared   # Partition to submit to
#SBATCH -N 1     # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 4     # cores requested
#SBATCH --mem=2000  # memory in Mb
#SBATCH -o hist_outfile_hist  # send stdout to outfile (with job ID)
#SBATCH -e hist_errfile_hist  # send stderr to errfile (with job ID)
#SBATCH -t 24:00:00  # time requested in hour:minute:second

# List of embeddings
embeddings=("esm1" "esm2" "esm3" "esm3_coord")

# Loop through each embedding and submit a separate job for each
for embed in "${embeddings[@]}"; do
    # Submit each embedding run as a separate job
    sbatch <<EOF
#!/bin/sh
#SBATCH -p shakhnovich,shared   # Partition to submit to
#SBATCH -N 1     # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 4     # cores requested
#SBATCH --mem=2000  # memory in Mb
#SBATCH -o hist_outfile_hist  # send stdout to outfile (with job ID)
#SBATCH -e hist_errfile_hist  # send stderr to errfile (with job ID)
#SBATCH -t 24:00:00  # time requested in hour:minute:second
export LD_LIBRARY_PATH=/n/home00/mhuot/.conda/envs/lantern/lib:$LD_LIBRARY_PATH
source activate lantern
cd ../plots

# Run the Python script with the specified embedding
echo "Running hist_al.py with embed=${embed}"
python3 hist_al.py --embed "$embed"

EOF
done
