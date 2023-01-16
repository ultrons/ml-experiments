NCORE=256
ZONE=us-central2-b
RUNTIME_VERSION=tpu-vm-v4-base
PREFIX=sivaibhav-exp-test
EXECUTE_LEVEL=4
#PROJECT_ID=mlperf-high-priority-project
PROJECT_ID=tpu-prod-env-one-vm

while getopts n:z:r:p:e: flag
do
    case "${flag}" in
        n) NCORE=${OPTARG};;
        z) ZONE=${OPTARG};;
        r) RUNTIME_VERSION=${OPTARG};;
        p) PREFIX=${OPTARG};;
        e) EXECUTE_LEVEL=${OPTARG};;
    esac
done

#TPU_NAME=${PREFIX}-${NCORE}
TPU_NAME=dontdelete-256
ACCELERATOR_TYPE=v4-$NCORE

SCOPES=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/pubsub

# Create Instance
if [ $EXECUTE_LEVEL -eq 0 ] || [ 0 -lt $EXECUTE_LEVEL  ]; then
    gcloud compute tpus tpu-vm create ${TPU_NAME} \
    --zone ${ZONE} \
    --project ${PROJECT_ID} \
    --accelerator-type ${ACCELERATOR_TYPE} \
    --version ${RUNTIME_VERSION} \
    --scopes=$SCOPES
fi

if [ $EXECUTE_LEVEL -eq -5 ] || [ 1 -lt $EXECUTE_LEVEL  ]; then
    gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME}  --zone ${ZONE} \
    --worker all \
    --command "
sudo pkill python
sudo lsof -w /dev/accel0 | grep /dev/accel0 | awk '{print \"sudo kill -9 \" \$2}' | sort | uniq | sh
sudo rm -f /tmp/libtpu_lockfile
sudo mkdir -p /tmp/tpu_logs && sudo chmod a+w -R /tmp/tpu_logs
"
fi

# Setup environment
if [ $EXECUTE_LEVEL -eq -1 ] || [ 1 -lt $EXECUTE_LEVEL  ]; then
    gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME}  --zone ${ZONE} \
    --worker all \
    --command "
    pip3 install https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch-nightly-cp38-cp38-linux_x86_64.whl --user
    pip3 install  https://storage.googleapis.com/tpu-pytorch/wheels/tpuvm/torch_xla-nightly-cp38-cp38-linux_x86_64.whl --user
    sudo pip uninstall -y libtpu-nightly
    pip3 install torch_xla[tpuvm] --user
    cd ~
git clone -b fsdp-exp  https://github.com/ultrons/transformers.git
cd transformers
pip3 install -e .
pip3 install datasets --user
pip3 install evaluate --user
pip3 install scikit-learn --user

"
fi


# Launch Training
if [ $EXECUTE_LEVEL -eq -2 ] || [ 2 -lt $EXECUTE_LEVEL  ]; then
    gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME}  --zone ${ZONE} \
    --worker all \
    --command "
    rm -rf xla_add
    git clone https://github.com/ultrons/xla_add.git -b temp-hack
    cd xla_add
    pip install -e .
"
fi

# Launch Training
if [ $EXECUTE_LEVEL -eq -3 ] || [ 3 -lt $EXECUTE_LEVEL  ]; then
    gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME}  --zone ${ZONE} \
    --worker all \
    --command "
cd transformers
git pull --rebase
screen -d -m bash -c \"bash ./train-51b.sh\"
"
fi

if [ $EXECUTE_LEVEL -eq -4 ] ; then
    gcloud alpha compute tpus tpu-vm delete ${TPU_NAME}  --zone ${ZONE}
fi
#echo \"#export LIBTPU_INIT_ARGS=\\\"--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_use_barna_core_for_offloading=true --xla_tpu_enable_latency_hiding_scheduler=true\\\"\" >> run.sh
##screen -d -m bash -c \"export EXP_PREFIX=32b-remat-12; export XLA_FLAGS=\"--xla_jf_spmd_threshold_for_windowed_einsum_mib=0,--xla_tpu_spmd_threshold_for_allgather_cse=10000,--xla_tpu_spmd_rewrite_einsum_with_reshape=true,--xla_enable_async_all_gather=true\"; bash ./sample-train-decode-32b.sh &> train.log\"
#screen -d -m bash -c \"export EXP_PREFIX=32b-2048; export XLA_FLAGS=\"--xla_dump_to=./xla.d\"; bash ./sample-train-decode-32b.sh &> train.log\"
#xla_jf_spmd_threshold_for_windowed_einsum_mib=0,xla_tpu_spmd_threshold_for_allgather_cse=10000,xla_tpu_spmd_rewrite_einsum_with_reshape=true,xla_enable_async_all_gather=true,jax_use_barna_core_for_offloading=true,xla_tpu_enable_latency_hiding_scheduler=true
