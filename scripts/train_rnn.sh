#!/bin/bash -e

########################################################################################################################

# directories
if [ $(uname -s) == "Darwin" ]; then
  if [ $USER == "ahmad" ]; then
    scriptsdir='/Users/ahmad/software/snaplab_github/neuro_rnn/scripts'
    datadir='/Users/ahmad/software/snaplab_github/neuro_rnn/data'
    outdir='/Users/ahmad/data/rutgers/neuro_rnn/results/pytorch/model'
  fi
else
  if [ $USER == "lindenmp" ]; then
    scriptsdir='/home/lindenmp/research_projects/neuro_rnn/scripts'
    datadir='/home/lindenmp/research_projects/neuro_rnn/data'
    outdir='/media/lindenmp/storage_ssd/research_projects/neuro_rnn/results/model_cpu'
  fi
fi

# path to inputs tsv
params_file="$datadir/model_params.csv" # < < < < < < < < < < < < < < < < SELECT MODELS FILE HERE

# path to log file
log_file="$outdir/run_training_$(date '+%Y-%m-%d-%H-%M-%S').log"

[ ! -d "$outdir" ] && mkdir -p "$outdir"

# rnn log settings
epoch_log=100

# device settings
device='cpu'
n_threads=12
if [ ${device} == 'cpu' ] && [ ${n_threads} -gt 1 ]; then
  echo "suspending all cuda devices"
  export CUDA_VISIBLE_DEVICES=""
  export OMP_NUM_THREADS=${n_threads}
  export MKL_NUM_THREADS=${n_threads}
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


########################################################################################################################

########################################################################################################################

{ # start of logging code block

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
    
    echo
    echo "================================================================================================================"
    echo
    echo "================================================================================================================"
    echo 
    echo "Model ...... $((line_index-1))"
    echo "Task ....... $task"
    echo "Kernel ..... $kernel_type"
    echo "Masks ...... $mask_weights"
    echo "Lambda ..... $reg_weight"
    echo "Epochs ..... $n_epochs"
    echo "Runs ....... $n_runs"
    echo "Started .... $(date '+%Y-%m-%d-%H-%M-%S')"

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
      --device ${device} \
      --n_threads ${n_threads}  2>&1 # capture stdout and stderr 

    echo 
    echo "Finished ... $(date '+%Y-%m-%d-%H-%M-%S')"

  done < "$params_file"

  echo
  echo "================================================================================================================"
  echo
  echo "================================================================================================================"
  echo 

} | tee -a "$log_file" # end of logging code block

########################################################################################################################

