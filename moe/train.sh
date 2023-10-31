export LIBTPU_INIT_ARGS="--xla_enable_async_all_gather=true"
export ACCELERATOR_TYPE=$(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/accelerator-type -H "Metadata-Flavor: Google")
export FLAX_PROFILE=1
export MODEL_DIR="gs://sivaibhav-exp/t5x/t5x-models/test/${ACCELERATOR_TYPE}-${EXP_PREFIX:=scale-ie}"
export XLA_FLAGS="--xla_dump_to=./xla.d"
python3 $HOME/t5x/t5x/train.py \
  --gin_search_paths="$HOME/flaxformer"  \
  --gin_file=$HOME/my_config.gin \
  --gin_file=$HOME/flaxformer/flaxformer/t5x/configs/moe/models/switch_base.gin \
  --gin.MODEL_PARALLEL_SUBMESH="[2,1,1,1]" \
  --gin.NUM_ENCODER_SPARSE_LAYERS=2 \
  --gin.NUM_DECODER_SPARSE_LAYERS=2 \
  --gin.MIXTURE_OR_TASK_NAME="'wikipedia_20190301.en_v003_unsupervised'" \
  --gin.MIXTURE_OR_TASK_MODULE="'t5.data.tasks'" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 1024, 'targets': 1024}" \
  --gin.TRAIN_STEPS=60 \
  --gin.DROPOUT_RATE=0 \
  --alsologtostderr \
