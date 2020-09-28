import kfp
import kfp.components as comp
import kfp.dsl as dsl
from os import path
import json
import yaml

cs = comp.ComponentStore()

component_path = path.join(path.dirname(__file__), 'steps')
cs.local_search_paths.append(component_path)
caip_train_op = comp.load_component_from_url(
            'https://raw.githubusercontent.com/kubeflow/pipelines/1.0.0/'
            'components/gcp/ml_engine/train/component.yaml')
#pre_process_op = cs.load_component('preProcess')
param_comp = cs.load_component('get_tuned_params')

# Config parameters
PROJECT_ID = 'pytorch-tpu-nfs'
REGION = 'us-central1'
FAIRSEQ_IMAGE = 'gcr.io/pytorch-tpu-nfs/fairseq-lm'

training_input_json = './steps/hypertune/config.yaml'
with open(training_input_json) as f:
    training_input = json.dumps(yaml.safe_load(f)['trainingInput'])

pipeline_args = {
    'project_id': PROJECT_ID,
    'region': REGION,
    'args': json.dumps([
        '--task', 'language_modeling',
        'data-bin/wikitext-103',
        '--save-dir', 'checkpoints/transformer_wikitext-103',
        '--arch', 'transformer_lm', '--share-decoder-input-output-embed',
        '--dropout', '0.1',
        '--optimizer', 'adam', '--adam-betas', '(0.9, 0.98)',
        '--clip-norm', '0.0',
        '--lr-scheduler', 'inverse_sqrt',
        '--warmup-updates', '4000',
        '--warmup-init-lr', '1e-07',
        '--tokens-per-sample', '512',
        '--sample-break-mode', 'none',
        '--max-tokens', '1024',
        '--update-freq', '16',
        '--fp16',
        '--max-update', '500',
    ]),
    'master_image_uri': FAIRSEQ_IMAGE,
    'training_input': training_input,
    'job_id_prefix': '',
    'job_id': '',
    'wait_interval': '30'
        }

@dsl.pipeline(
        name='KFP-Pipelines Example',
        description='Kubeflow pipeline using pre-built components'
        )
def pipeline(
    project_id=PROJECT_ID,
    region=REGION,
    args=json.dumps([
        '--task', 'language_modeling',
        'data-bin/wikitext-103',
        '--save-dir', 'checkpoints/transformer_wikitext-103',
        '--arch', 'transformer_lm', '--share-decoder-input-output-embed',
        '--dropout', '0.1',
        '--optimizer', 'adam', '--adam-betas', '(0.9, 0.98)',
        '--clip-norm', '0.0',
        '--lr-scheduler', 'inverse_sqrt',
        '--warmup-updates', '4000',
        '--warmup-init-lr', '1e-07',
        '--tokens-per-sample', '512',
        '--sample-break-mode', 'none',
        '--max-tokens', '1024',
        '--update-freq', '16',
        '--fp16',
        '--max-update', '500',
    ]),
    master_image_uri=FAIRSEQ_IMAGE,
    training_input=training_input,
    job_id_prefix='',
    job_id='',
    wait_interval='30'
):
    """ Pipeline (DAG) definition """

    hypertune = caip_train_op(
        project_id=project_id,
        region=region,
        args=args,
        master_image_uri=master_image_uri,
        training_input=training_input,
        job_id_prefix=job_id_prefix,
        job_id=job_id,
        wait_interval=wait_interval)


client = kfp.Client(host='321ff3bfe3fa6d70-dot-us-central2.pipelines.'
                    'googleusercontent.com')
#client.create_run_from_pipeline_func(pipeline, pipeline_args)
