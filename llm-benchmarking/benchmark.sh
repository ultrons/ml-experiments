NCORE=128
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
#sudo gsutil -m  cp -r gs://sivaibhav-exp/_libtpu.so /lib/libtpu.so
#sudo mv /lib/libtpu.so /lib/libtpu.so.orig
#sudo gsutil -m cp -r gs://sivaibhav-exp/libtpu-xprof.so /lib/libtpu.so
rm -rf t5x
git clone https://github.com/ultrons/t5x.git -b profile-study
cd t5x
pip install -e . -f   https://storage.googleapis.com/jax-releases/libtpu_releases.html
cd ~
git clone https://github.com/google/CommonLoopUtils.git
pip uninstall -y clu
cd CommonLoopUtils
pip install -e .
cd ~
pip uninstall -y jax jaxlib
pip install jax[tpu]==0.3.25 -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install tb-nightly tbp-nightly tf-nightly tensorflow-text-nightly
pip install --upgrade orbax==0.0.15
"
fi

# Test if all the devices are detected
if [ $EXECUTE_LEVEL -eq -2 ] || [ 2 -lt $EXECUTE_LEVEL  ]; then
    gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME}  --zone ${ZONE} \
    --worker all \
    --command "
python3 -c \"import jax; print(jax.devices())\" | tee  ~/jax-device-discovery.log
"
fi

# Launch Training
if [ $EXECUTE_LEVEL -eq -3 ] || [ 3 -lt $EXECUTE_LEVEL  ]; then
    gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME}  --zone ${ZONE} \
    --worker all \
    --command "
cd t5x
git pull --rebase
echo \"export EXP_PREFIX=40b-2d-2d-partition-$$\" > run.sh
screen -d -m bash -c \"source ./run.sh; bash ./sample-train-decode-40b.sh &> train.log\"
"
fi

if [ $EXECUTE_LEVEL -eq -4 ] ; then
    gcloud alpha compute tpus tpu-vm delete ${TPU_NAME}  --zone ${ZONE}
fi
#echo \"#export LIBTPU_INIT_ARGS=\\\"--xla_jf_spmd_threshold_for_windowed_einsum_mib=0 --xla_tpu_spmd_threshold_for_allgather_cse=10000 --xla_tpu_spmd_rewrite_einsum_with_reshape=true --xla_enable_async_all_gather=true --jax_use_barna_core_for_offloading=true --xla_tpu_enable_latency_hiding_scheduler=true\\\"\" >> run.sh
##screen -d -m bash -c \"export EXP_PREFIX=32b-remat-12; export XLA_FLAGS=\"--xla_jf_spmd_threshold_for_windowed_einsum_mib=0,--xla_tpu_spmd_threshold_for_allgather_cse=10000,--xla_tpu_spmd_rewrite_einsum_with_reshape=true,--xla_enable_async_all_gather=true\"; bash ./sample-train-decode-32b.sh &> train.log\"
#screen -d -m bash -c \"export EXP_PREFIX=32b-2048; export XLA_FLAGS=\"--xla_dump_to=./xla.d\"; bash ./sample-train-decode-32b.sh &> train.log\"
#xla_jf_spmd_threshold_for_windowed_einsum_mib=0,xla_tpu_spmd_threshold_for_allgather_cse=10000,xla_tpu_spmd_rewrite_einsum_with_reshape=true,xla_enable_async_all_gather=true,jax_use_barna_core_for_offloading=true,xla_tpu_enable_latency_hiding_scheduler=true
