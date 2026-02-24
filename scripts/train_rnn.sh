#!/bin/bash -e

########################################################################################################################

params="202602"

# directories
if [ $(uname -s) == "Darwin" ]; then
  if [ $USER == "ahmad" ]; then
    scriptsdir='/Users/ahmad/software/snaplab_github/neuro_rnn/scripts'
    datadir='/Users/ahmad/software/snaplab_github/neuro_rnn/data'
    outdir="/Users/ahmad/data/rutgers/neuro_rnn/results/pytorch/model/${params}"
  fi
else
  if [ $USER == "lindenmp" ]; then
    scriptsdir='/home/lindenmp/research_projects/neuro_rnn/scripts'
    datadir='/home/lindenmp/research_projects/neuro_rnn/data'
    outdir='/media/lindenmp/storage_ssd/research_projects/neuro_rnn/results/model_cpu'
  elif [ $USER == "ab2792" ]; then
    scriptsdir='/home/ab2792/software/snaplab_github/neuro_rnn/scripts'
    datadir='/home/ab2792/software/snaplab_github/neuro_rnn/data'
    outdir="/home/ab2792/data/neuro_rnn/results/pytorch/model/${params}"
  fi
fi

# activate conda env
source ~/.bashrc
conda activate neuro_rnn

# path to inputs csv
params_file="$datadir/model_params_${params}.csv" # < < < < < < < < < < < < < < < < SELECT MODELS FILE HERE
# tmp_params_file="/tmp/$(basename ${params_file%.csv})_${RANDOM}${RANDOM}.csv"
# cp "$params_file" "$tmp_params_file"
# params_file="$tmp_params_file"

[ ! -d "$outdir" ] && mkdir -p "$outdir"

# rnn log settings
epoch_log=100

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

# optional: pass space-separated 0-based model indices to run a subset (e.g. "0 2 5")
if [ "$1" != "" ]; then
  selected_indices="$1"
else
  selected_indices=""
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

  log_file="$outdir/model-${model_index}_$(date '+%Y-%m-%d-%H-%M-%S').log"

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
    --epoch_log ${epoch_log} \
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
