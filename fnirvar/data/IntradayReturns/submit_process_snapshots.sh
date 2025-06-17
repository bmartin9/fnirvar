#PBS -N backtesting 
#PBS -l walltime=03:30:00 
#PBS -l select=1:ncpus=1:mem=8gb

module load anaconda3/personal
source activate fnirvar

cd $PBS_O_WORKDIR
python process_snapshot.py --all
