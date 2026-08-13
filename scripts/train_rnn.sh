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
# scriptsdir is this script's own location; datadir/outdir come from the project
# path config (see paths.yaml.template), and can be overridden by exporting
# NEURO_RNN_DATA_DIR / NEURO_RNN_MODEL_DIR.
scriptsdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repodir="$(dirname "$scriptsdir")"

# activate conda env (skip if you manage the environment yourself)
# NOTE: this must happen before the path queries below, so that `python` is the
# project interpreter.
if [ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]; then
  # Source first: `conda` is a shell function that does not exist until the
  # shell is initialised, so calling it beforehand aborts the script (set -e).
  source ~/.bashrc
  # Drop any already-active env so environments do not stack. Tolerate failure:
  # there may be nothing to deactivate.
  conda deactivate 2>/dev/null || true
  conda activate neuro_rnn
fi

datadir=$(cd "$repodir" && python -m src.config data_dir)
outdir=$(cd "$repodir" && python -m src.config model_dir --params_name "$params_name")

[ ! -d "$outdir" ] && mkdir -p "$outdir"

# rnn log settings
print_freq=100
log_freq=100
write_freq=1000

# device settings
device=${DEVICE:-cpu} # 'cpu' or 'cuda' or 'mps'
n_threads=${N_THREADS:-4}
if [ ${device} == 'cpu' ] && [ ${n_threads} -gt 1 ]; then
  echo "suspending all cuda devices"
  export CUDA_VISIBLE_DEVICES=""
  export OMP_NUM_THREADS=${n_threads}
  export MKL_NUM_THREADS=${n_threads}
elif [ ${device} != 'cpu' ]; then
  n_threads=1
fi

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
