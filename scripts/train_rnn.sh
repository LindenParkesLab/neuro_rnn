#!/bin/bash -e

########################################################################################################################

usage(){
cat << EOF

Usage: $(basename $0) <params_csv_file> [index1 index2 ...]

EOF
exit 1
}

[ $# -eq 0 ] && usage

# first positional argument: path to model params CSV file (required)
if [ ! -f "$1" ]; then
  echo "$(basename $0): Error: First argument must be a valid path to a model params CSV file."
  exit 1
fi
params_file="$1"
params_name=$(basename "${1%.csv}")
shift

# remaining positional arguments: optional space-separated 0-based model indices (e.g. 0 2 5)
selected_indices="$*"

# directories
if [ $(uname -s) == "Darwin" ]; then
  if [ $USER == "ahmad" ]; then
    scriptsdir='/Users/ahmad/software/snaplab_github/neuro_rnn/scripts'
    datadir='/Users/ahmad/software/snaplab_github/neuro_rnn/data'
    outdir="/Users/ahmad/data/rutgers/neuro_rnn/results/pytorch/model/${params_name}"
  fi
else
  if [ $USER == "lindenmp" ]; then
    scriptsdir='/home/lindenmp/research_projects/neuro_rnn/scripts'
    datadir='/home/lindenmp/research_projects/neuro_rnn/data'
    outdir='/media/lindenmp/storage_ssd/research_projects/neuro_rnn/results/model_cpu'
  elif [ $USER == "ab2792" ]; then
    scriptsdir='/home/ab2792/software/snaplab_github/neuro_rnn/scripts'
    datadir='/home/ab2792/software/snaplab_github/neuro_rnn/data'
    outdir="/home/ab2792/data/neuro_rnn/results/pytorch/model/${params_name}"
  fi
fi

# activate conda env
source ~/.bashrc
conda activate neuro_rnn

[ ! -d "$outdir" ] && mkdir -p "$outdir"

# rnn log settings
print_freq=100
log_freq=100
write_freq=1000

# device settings
device='cpu' # 'cpu' or 'cuda' or 'mps'
n_threads=10
if [ ${device} == 'cpu' ] && [ ${n_threads} -gt 1 ]; then
  echo "suspending all cuda devices"
  export CUDA_VISIBLE_DEVICES=""
  export OMP_NUM_THREADS=${n_threads}
  export MKL_NUM_THREADS=${n_threads}
elif [ ${device} != 'cpu' ]; then
  n_threads=1
fi

########################################################################################################################

# # clean csv to avoid encoding issues
# python ${scriptsdir}/clean_csv.py "$params_file"
# params_file="${params_file%.csv}_clean.csv"

########################################################################################################################

model_index=0
while IFS= read -r line || [ -n "$line" ]; do

  if [ "$selected_indices" != "" ] && [ "$(echo $selected_indices | grep -w $model_index)" == "" ]; then
    model_index=$((model_index + 1))
    continue
  fi

  log_file="$outdir/model-$(printf '%03i' ${model_index})_$(date '+%Y-%m-%d-%H-%M-%S').log"

  { # start of logging code block

  echo
  echo "================================================================================================================"
  echo
  echo "================================================================================================================"
  echo
  echo "Input model index ......... $model_index"
  echo "Date started .............. $(date '+%Y-%m-%d-%H-%M-%S')"
  echo

  python ${scriptsdir}/train_rnn.py \
    --params_csv ${params_file} \
    --model_index ${model_index} \
    --datadir ${datadir} \
    --outdir ${outdir} \
    --print_freq ${print_freq} \
    --log_freq ${log_freq} \
    --write_freq ${write_freq} \
    --device ${device} \
    --n_threads ${n_threads} 2>&1 # capture stdout and stderr

  echo
  echo "Date finished ... $(date '+%Y-%m-%d-%H-%M-%S')"
  echo
  echo "================================================================================================================"
  echo
  echo "================================================================================================================"
  echo

  } | tee -a "$log_file" # end of logging code block

  model_index=$((model_index + 1))

done < <(tail -n +2 "$params_file")

########################################################################################################################
