#PBS -N backtesting 
#PBS -l walltime=03:30:00 
#PBS -l select=1:ncpus=173:mem=8gb
#PBS -J 1-173

module load anaconda3/personal
source activate fnirvar

MODEL_DIR=~/phd/projects/factors/fnirvar/models/IntradayReturns/30minutely
SNAP_LIST=~/phd/projects/factors/fnirvar/fnirvar/data/IntradayReturns/month_list.txt 

MONTH=$(sed -n "${PBS_ARRAY_INDEX}p" ${SNAP_LIST})

cd $PBS_O_WORKDIR
python backtest.py --first "${MONTH}" --last "${MONTH}"