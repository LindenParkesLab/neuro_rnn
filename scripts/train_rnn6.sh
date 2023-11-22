scriptsdir='/home/lindenmp/research_projects/neuro_rnn/scripts'
outdir='/media/lindenmp/storage_ssd/research_projects/neuro_rnn/results/pytorch/model'

task='MultiSensoryIntegration-v0'
seq_len_base=11
standardize_task='False'
kernel_type='None'

rnn_model='rnn-tanh'
hidden_size=100
n_runs=50
n_epochs=30000
reg_type='l2'

for mask_weights in 'True' 'False'; do
  for reg_weight in 0.0 0.0001 0.001 0.002 0.003; do
    for decision in 400 300 200; do
      seq_len=$(( seq_len_base + ((decision-100)/100) ))

      python ${scriptsdir}/train_rnn.py --task ${task} --rnn_model ${rnn_model} --kernel_type ${kernel_type} \
      --mask_weights ${mask_weights} --hidden_size ${hidden_size} --n_runs ${n_runs} --n_epochs ${n_epochs} \
      --decision ${decision} --seq_len ${seq_len} --outdir ${outdir} --reg_weight ${reg_weight} --reg_type ${reg_type} \
      --standardize_task ${standardize_task}
    done
  done
done
