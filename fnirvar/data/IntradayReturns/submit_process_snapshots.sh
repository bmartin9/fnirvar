#PBS -N backtesting 
#PBS -l walltime=00:30:00 
#PBS -l select=1:ncpus=172:mem=8gb
#PBS -J 1-172


module load anaconda3/personal
source activate fnirvar

cd $PBS_O_WORKDIR
MONTH=$(ls snapshots | sort | sed -n "$((PBS_ARRAY_INDEX+1))p")
python process_snapshot.py --snapshot_dir snapshots/$MONTH

