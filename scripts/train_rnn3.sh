scriptsdir='/home/lindenmp/research_projects/neuro_rnn/scripts'
outdir='/media/lindenmp/storage_ssd/research_projects/neuro_rnn/results/pytorch/model'

task='PerceptualDecisionMaking-v0'
seq_len_base=22
standardize_task='False'
kernel_type='None'

seq_len_multi=5
rnn_model='rnn-tanh'
hidden_size=100
batch_size=32
n_runs=25
n_epochs=30000
reg_type='l2'

for mask_weights in 'True' 'False'; do
  for reg_weight in 0.0 0.0001 0.001 0.0015 0.002 0.003; do
    for decision in 400 300 200 100; do
      seq_len=$(( seq_len_base + ((decision-100)/100) ))
      seq_len=$(( seq_len * seq_len_multi ))

      python ${scriptsdir}/train_rnn.py --outdir ${outdir} --task ${task} --seq_len ${seq_len} --standardize_task ${standardize_task} --kernel_type ${kernel_type} \
      --rnn_model ${rnn_model} --hidden_size ${hidden_size} --batch_size ${batch_size} --n_runs ${n_runs} --n_epochs ${n_epochs} --reg_type ${reg_type} \
      --mask_weights ${mask_weights} --reg_weight ${reg_weight} --decision ${decision}
    done
  done
done
