# Install packages
pip install --upgrade pip
pip install --upgrade setuptools

# utils
conda install ipython tqdm numpy scipy pandas scikit-learn scikit-learn-intelex matplotlib seaborn jupyterlab \
ipywidgets networkx numba pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install gym nctpy

#git clone https://github.com/netneurolab/conn2res.git ~/research_projects/neuro_rnn/packages/conn2res
#cd ~/research_projects/neuro_rnn/packages/conn2res
#pip install .

git clone https://github.com/gyyang/neurogym.git ~/research_projects/neuro_rnn/packages/neurogym
cd ~/research_projects/neuro_rnn/packages/neurogym
pip install -e .

git clone https://github.com/aestrivex/bctpy.git ~/research_projects/neuro_rnn/packages/bctpy
cd ~/research_projects/neuro_rnn/packages/bctpy
pip install .

# export
cd ~/research_projects/neuro_rnn
conda env export > environment.yml --no-builds

python setup.py install