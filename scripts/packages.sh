# install packages: linux

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

# install packages: macbook
project_dir='/Users/ahmad/software/snaplab_github/neuro_rnn'
pip install --upgrade pip
pip install --upgrade setuptools

conda install ipython tqdm numpy scipy pandas scikit-learn matplotlib seaborn jupyterlab \
ipywidgets networkx numba pytorch::pytorch torchvision torchaudio -c pytorch
pip install --upgrade notebook
pip install gym nctpy

git clone https://github.com/gyyang/neurogym.git ${project_dir}/packages/neurogym
cd ${project_dir}/packages/neurogym
pip install -e .

git clone https://github.com/aestrivex/bctpy.git ${project_dir}/packages/bctpy
cd ${project_dir}/packages/bctpy
pip install .

cd ${project_dir}
python setup.py install
