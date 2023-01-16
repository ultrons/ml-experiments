NCORE=32
ZONE=us-central2-b
RUNTIME_VERSION=v2-alpha-tpuv4
PREFIX=sivaibhav-exp-test
EXECUTE_LEVEL=4

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

TPU_NAME=${PREFIX}-${NCORE}
ACCELERATOR_TYPE=v4-$NCORE

SCOPES=https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append,https://www.googleapis.com/auth/pubsub

# Create Instance
if [ $EXECUTE_LEVEL -eq 0 ] || [ 0 -lt $EXECUTE_LEVEL  ]; then
    gcloud alpha compute tpus tpu-vm create ${TPU_NAME} \
    --zone ${ZONE} \
    --accelerator-type ${ACCELERATOR_TYPE} \
    --version ${RUNTIME_VERSION} \
    --scopes=$SCOPES
fi

# Setup environment
if [ $EXECUTE_LEVEL -eq -1 ] || [ 1 -lt $EXECUTE_LEVEL  ]; then
    gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME}  --zone ${ZONE} \
    --worker all \
    --command "
sudo gsutil -m  cp -r gs://sivaibhav-exp/_libtpu.so /lib/libtpu.so
sudo pip uninstall -y tf-nightly
sudo pip uninstall -y tb-nightly
sudo pip uninstall -y keras-nightly
sudo pip uninstall -y jax
sudo pip uninstall -y jaxlib
pip install -U tensorboard-plugin-profile
rm -rf t5x
git clone https://github.com/ultrons/t5x.git -b benchmark
cd t5x
python3 -m pip install -e . -f   https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip uninstall -y libtpu-nightly
gsutil cp gs://cloud-tpu-tpuvm-v4-artifacts/wheels/libtpu/latest/libtpu_tpuv4-0.1.dev* .
pip install libtpu_tpuv4-0.1.dev*
"
fi

# Test if all the devices are detected
if [ $EXECUTE_LEVEL -eq -2 ] || [ 2 -lt $EXECUTE_LEVEL  ]; then
    gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME}  --zone ${ZONE} \
    --worker all \
    --command "
python3 -c \"import jax; print(jax.devices())\" > ~/jax-device-discovery.log
"
fi

# Launch Training
if [ $EXECUTE_LEVEL -eq -3 ] || [ 3 -lt $EXECUTE_LEVEL  ]; then
    gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME}  --zone ${ZONE} \
    --worker all \
    --command "
cd t5x
git pull --rebase
screen -d -m bash -c \"bash ./sample-train-xxl-remat.sh &> train.log\"
"
fi
if [ $EXECUTE_LEVEL -eq -4 ] ; then
    gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME}  --zone ${ZONE} \
    --worker all \
    --command "
pkill python
rm -rf /tmp/libtpu*
"
fi
