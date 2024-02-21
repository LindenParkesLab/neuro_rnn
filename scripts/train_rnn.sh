########################################################################################################################
# directories
#scriptsdir='/home/lindenmp/research_projects/neuro_rnn/scripts'
#outdir='/media/lindenmp/storage_ssd/research_projects/neuro_rnn/results/pytorch/model'
scriptsdir='/Users/ahmad/software/snaplab_github/neuro_rnn/scripts'
datadir='/Users/ahmad/software/snaplab_github/neuro_rnn/data'
outdir='/Users/ahmad/data/rutgers/neuro_rnn/results/pytorch/model'
########################################################################################################################

########################################################################################################################
# settings
seq_len_multi=5
rnn_model='rnn-tanh'
hidden_size=100
batch_size=32
n_runs=12 #100
n_epochs=2500 #25000
epoch_log=100
reg_type='l2'

mask_weights='True'
reg_weight=0.002
decision=400
########################################################################################################################

########################################################################################################################
# run training
for task in 'PerceptualDecisionMaking-v0' 'MultiSensoryIntegration-v0' 'ContextDecisionMaking-v0'; do
  if [ ${task} == 'PerceptualDecisionMaking-v0' ]; then
    seq_len_base=22
    standardize_task='False'
  elif [ ${task} == 'MultiSensoryIntegration-v0' ]; then
    seq_len_base=11
    standardize_task='False'
  elif [ ${task} == 'ContextDecisionMaking-v0' ]; then
    seq_len_base=13
    standardize_task='True'
  fi

  for kernel_type in 'sa_axis' 'euclidean' 'None'; do
    seq_len=$(( seq_len_base + ((decision-100)/100) ))
    seq_len=$(( seq_len * seq_len_multi ))

    python ${scriptsdir}/train_rnn.py \
    --outdir ${outdir} \
    --datadir ${datadir} \
    --task ${task} \
    --seq_len ${seq_len} \
    --standardize_task ${standardize_task} \
    --decision ${decision} \
    --mask_weights ${mask_weights} \
    --kernel_type ${kernel_type} \
    --reg_weight ${reg_weight} \
    --rnn_model ${rnn_model} \
    --hidden_size ${hidden_size} \
    --batch_size ${batch_size} \
    --n_runs ${n_runs} \
    --n_epochs ${n_epochs} \
    --epoch_log ${epoch_log} \
    --reg_type ${reg_type}
  done
done
########################################################################################################################

