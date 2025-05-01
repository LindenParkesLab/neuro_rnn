#!/bin/bash -e

########################################################################################################################

# directories
if [ $(uname -s) == "Darwin" ]; then
  if [ $USER == "ahmad" ]; then
    scriptsdir='/Users/ahmad/software/snaplab_github/neuro_rnn/scripts'
    datadir='/Users/ahmad/software/snaplab_github/neuro_rnn/data'
    outdir='/Users/ahmad/data/rutgers/neuro_rnn/results/pytorch/model/20250430'
  fi
else
  if [ $USER == "lindenmp" ]; then
    scriptsdir='/home/lindenmp/research_projects/neuro_rnn/scripts'
    datadir='/home/lindenmp/research_projects/neuro_rnn/data'
    outdir='/media/lindenmp/storage_ssd/research_projects/neuro_rnn/results/model_cpu'
  elif [ $USER == "ab2792" ]; then
    scriptsdir='/home/ab2792/software/snaplab_github/neuro_rnn/scripts'
    datadir='/home/ab2792/software/snaplab_github/neuro_rnn/data'
    outdir='/home/ab2792/data/neuro_rnn/results/pytorch/model/20250203'
  fi
fi

# activate conda env
source ~/.bashrc
conda activate neuro_rnn

# path to inputs csv
params_file="$datadir/model_params_gng_pdm_dms.csv" # < < < < < < < < < < < < < < < < SELECT MODELS FILE HERE
tmp_params_file="/tmp/$(basename ${params_file%.csv})_${RANDOM}${RANDOM}.csv"
cp "$params_file" "$tmp_params_file" 
params_file="$tmp_params_file"

## path to log file
#log_file="$outdir/run_training_$(date '+%Y-%m-%d-%H-%M-%S').log"

[ ! -d "$outdir" ] && mkdir -p "$outdir"

# rnn log settings
epoch_log=100
init_rnn_weights_min="-0.01"
init_rnn_weights_max="0.01"

# device settings
device='cpu' # 'cpu' or 'cuda' or 'mps'
n_threads=8
if [ ${device} == 'cpu' ] && [ ${n_threads} -gt 1 ]; then
  echo "suspending all cuda devices"
  export CUDA_VISIBLE_DEVICES=""
  export OMP_NUM_THREADS=${n_threads}
  export MKL_NUM_THREADS=${n_threads}
elif [ ${device} != 'cpu' ]; then
  n_threads=1
fi

# are we running all models?
if [ "$1" != "" ]; then
  selected_lines="$1"
else
  selected_lines=""
fi

########################################################################################################################

########################################################################################################################

# clean csv to avoid encoding issues
python ${scriptsdir}/clean_csv.py "$params_file"
params_file="${params_file%.csv}_clean.csv"

# get column header numbers from csv
get_col_num() {
  # get_col_num <line1> <col_name>
  c=0
  for col in $(echo "${1}" | sed 's/\r//g' | sed 's/,/\n/g'); do
    c=$((c+1))
    found=0
    if [ "$col" == "$2" ]; then
      echo $c
      found=1 
      break
    fi
  done
  if [ $found -eq 0 ]; then
    exit 1
  fi
}

line=$(head -1 "$params_file")
col_task=$(get_col_num "$line" 'task')
col_seq_len_multi=$(get_col_num "$line" 'seq_len_multi')
col_rnn_model=$(get_col_num "$line" 'rnn_model')
col_hidden_size=$(get_col_num "$line" 'hidden_size')
col_batch_size=$(get_col_num "$line" 'batch_size')
col_learning_rate=$(get_col_num "$line" 'learning_rate')
col_n_runs=$(get_col_num "$line" 'n_runs')
col_n_epochs=$(get_col_num "$line" 'n_epochs')
col_mask_weights=$(get_col_num "$line" 'mask_weights')
col_reg_type=$(get_col_num "$line" 'reg_type')
col_reg_weight=$(get_col_num "$line" 'reg_weight')
col_kernel_type=$(get_col_num "$line" 'kernel_type')
col_kernel_normalization=$(get_col_num "$line" 'kernel_normalization')
col_time_step=$(get_col_num "$line" 'time_step')
col_alpha=$(get_col_num "$line" 'alpha')


########################################################################################################################

########################################################################################################################

#{ # start of logging code block

  line_index=0

  while read line; do 
    
    line_index=$((line_index+1))

    [ $line_index -eq 1 ] && continue # skip header

    if [ "$selected_lines" != "" -a "$(echo $selected_lines | grep -w $line_index)" == "" ]; then
      continue
    fi

    task="$(echo $line | awk -v c=${col_task} -F ',' '{print $c}')"
    seq_len_multi=$(echo $line | awk -v c=${col_seq_len_multi} -F ',' '{print $c}')
    rnn_model="$(echo $line | awk -v c=${col_rnn_model} -F ',' '{print $c}')"
    hidden_size=$(echo $line | awk -v c=${col_hidden_size} -F ',' '{print $c}')
    batch_size=$(echo $line | awk -v c=${col_batch_size} -F ',' '{print $c}')
    learning_rate=$(echo $line | awk -v c=${col_learning_rate} -F ',' '{print $c}')
    n_runs=$(echo $line | awk -v c=${col_n_runs} -F ',' '{print $c}')
    n_epochs=$(echo $line | awk -v c=${col_n_epochs} -F ',' '{print $c}')
    mask_weights=$(echo $line | awk -v c=${col_mask_weights} -F ',' '{print $c}')
    reg_type=$(echo $line | awk -v c=${col_reg_type} -F ',' '{print $c}')
    reg_weight=$(echo $line | awk -v c=${col_reg_weight} -F ',' '{print $c}')
    kernel_type="$(echo $line | awk -v c=${col_kernel_type} -F ',' '{print $c}')"
    kernel_normalization="$(echo $line | awk -v c=${col_kernel_normalization} -F ',' '{print $c}')"
    time_step="$(echo $line | awk -v c=${col_time_step} -F ',' '{print $c}')"
    alpha="$(echo $line | awk -v c=${col_alpha} -F ',' '{print $c}')"
    
    # path to log file
    lfp1="$outdir/task-${task}"
    lfp2="_model-${rnn_model}-${hidden_size}-${batch_size}-${learning_rate}-${n_runs}-${n_epochs}"
    lfp3="_wmask-${mask_weights}"
    lfp4="_reg-${reg_type}-${reg_weight}-${kernel_type}-${kernel_normalization}"
    lfp5="_alpha-${alpha}"
    lfp6="_$(date '+%Y-%m-%d-%H-%M-%S').log"
    log_file="${lfp1}${lfp2}${lfp3}${lfp4}${lfp5}${lfp6}"

    { # start of logging code block

    echo
    echo "================================================================================================================"
    echo
    echo "================================================================================================================"
    echo 
    echo "Input model index ......... $((line_index-1))"
    echo "Task name ................. $task"
    echo "Kernel type ............... $kernel_type"
    echo "Kernel normalization ...... $kernel_normalization"
    echo "Kernel I/O masks .......... $mask_weights"
    echo "Regularization (lambda) ... $reg_weight"
    echo "Number of epochs .......... $n_epochs"
    echo "Number of runs ............ $n_runs"
    echo "Date started .............. $(date '+%Y-%m-%d-%H-%M-%S')"
    echo

    python ${scriptsdir}/train_rnn.py \
      --outdir ${outdir} \
      --datadir ${datadir} \
      --task ${task} \
      --seq_len_multi ${seq_len_multi} \
      --mask_weights ${mask_weights} \
      --kernel_type ${kernel_type} \
      --reg_weight ${reg_weight} \
      --rnn_model ${rnn_model} \
      --hidden_size ${hidden_size} \
      --batch_size ${batch_size} \
      --learning_rate ${learning_rate} \
      --n_runs ${n_runs} \
      --n_epochs ${n_epochs} \
      --epoch_log ${epoch_log} \
      --reg_type ${reg_type} \
      --kernel_normalization ${kernel_normalization} \
      --dt ${time_step} \
      --alpha ${alpha} \
      --init_rnn_weights "${init_rnn_weights_min}" "${init_rnn_weights_max}" \
      --device ${device} \
      --n_threads ${n_threads}  2>&1 # capture stdout and stderr 

    echo 
    echo "Date finished ... $(date '+%Y-%m-%d-%H-%M-%S')"
    echo
    echo "================================================================================================================"
    echo
    echo "================================================================================================================"
    echo 
    
    } | tee -a "$log_file" # end of logging code block

  done < "$params_file"

########################################################################################################################

