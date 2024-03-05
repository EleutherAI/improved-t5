sudo apt update && sudo apt upgrade -y
sudo apt install software-properties-common -y
sudo add-apt-repository ppa:deadsnakes/ppa -y

sudo add-apt-repository ppa:ubuntu-toolchain-r/test -y
sudo apt-get update && sudo apt-get install libstdc++6 -y && sudo apt-get install libc6 -y

sudo apt install python3.10 -y
sudo apt install python3.10-distutils -y
virtualenv env-t5x --python=python3.10
source env-t5x/bin/activate

# python -m pip install tensorflow-datasets==4.8.2

# # Install t5
# git clone https://github.com/google-research/text-to-text-transfer-transformer.git
# pushd text-to-text-transfer-transformer
# python -m pip install -e .
# popd

pip install git+https://github.com/google-research/text-to-text-transfer-transformer.git
pip install git+https://github.com/google/flaxformer.git

# git clone https://github.com/google/flaxformer.git
# pushd flaxformer
# python -m pip install -e .
# popd

#git clone https://github.com/lintangsutawika/t-zero.git
#pushd t-zero
#python -m pip install -e .[seqio_tasks]
#popd

# Download improved-t5
git clone https://github.com/EleutherAI/improved-t5.git
pushd improved-t5
python -m pip install -e .
popd

# python -m pip install jax==0.4.12
# # python -m pip install jax[tpu]==0.4.12
# python -m pip install jaxlib==0.4.12
python -m pip install datasets
# Install T5x
git clone --branch=main https://github.com/google-research/t5x
pushd t5x
python -m pip install -e '.[tpu]' -f \
  https://storage.googleapis.com/jax-releases/libtpu_releases.html
popd
# python -m pip install flax==0.7.0