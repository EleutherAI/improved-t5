pip3 uninstall -y -r <(pip freeze)
# sudo apt-get install wget
# pip3 install wget
export PATH=/home/lintangsutawika/.local/bin:${PATH}
pip3 install -U pip
pip3 install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html --force-reinstall
pip3 install tensorflow==2.9.1
rm libtpu_tpuv4-0.1.dev*
gsutil cp gs://cloud-tpu-tpuvm-v4-artifacts/wheels/libtpu/latest/libtpu_tpuv4-0.1.dev* .
pip3 install libtpu_tpuv4-0.1.dev*

python3 -c "import jax ; print(jax.device_count())"

sudo rm -rf ~/code
mkdir -p ~/code
cd ~/code

# Install t5 first
git clone https://github.com/google-research/text-to-text-transfer-transformer.git
pushd text-to-text-transfer-transformer
pip3 install -e .
popd
pip3 install git+https://github.com/google-research/text-to-text-transfer-transformer.git@main#egg=t5

rm -r architecture-objective
git clone https://github.com/EleutherAI/architecture-objective.git
pip3 install git+https://github.com/EleutherAI/architecture-objective.git@main#egg=t5x
pushd architecture-objective
pip3 install -e .
#git checkout add_AliBi
git checkout main
popd

pip3 install git+https://github.com/EleutherAI/CommonLoopUtils.git@tfdataiterator#egg=clu
git clone https://github.com/EleutherAI/CommonLoopUtils.git
pushd CommonLoopUtils
git pull
git checkout tfdataiterator
pip3 install -e .
popd

cd ~/code/CommonLoopUtils
pip3 install -e .
cd - 

git clone https://github.com/EleutherAI/FLAN.git
cd ~/code/FLAN
pip3 install -e .
cd -

rm -r orbax
pip3 install git+https://github.com/google/orbax.git@v0.0.12

cd ~/code/orbax
pip3 install -e .
cd -

git clone https://github.com/google-research/text-to-text-transfer-transformer.git
pushd text-to-text-transfer-transformer
pip3 install -e .
popd

rm -f /tmp/libtpu_lockfile
sudo rm -rf /tmp/tpu_logs/
