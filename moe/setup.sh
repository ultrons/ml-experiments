git clone https://github.com/ultrons/t5x -b temp-fix
cd t5x/
pip install -e .
cd $HOME
pip install  --use-deprecated=legacy-resolver  'flax @ git+https://github.com/google/flax#egg=flax' 
git clone https://github.com/ultrons/flaxformer -b fix-relative
cd flaxformer/
pip install -e .
cd $HOME
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install t5
