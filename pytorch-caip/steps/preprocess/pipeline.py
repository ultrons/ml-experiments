import kfp
import kfp.components as comp
import kfp.dsl as dsl
from os import path
import json
import yaml

cs = comp.ComponentStore()

component_path = path.join(path.dirname(__file__), '..')
cs.local_search_paths.append(component_path)
preprocess_op = cs.load_component('preprocess')

# Config parameters
PROJECT_ID = 'pytorch-tpu-nfs'
REGION = 'us-central1'
FAIRSEQ_IMAGE = 'gcr.io/pytorch-tpu-nfs/fairseq-lm'

common_args = json.dumps([
        '--task', 'language_modeling',
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
    ])

pipeline_args = {
    'project_id': PROJECT_ID,
    'region': REGION,
    'args': common_args,
    'master_image_uri': FAIRSEQ_IMAGE,
    'job_id_prefix': '',
    'job_id': '',
    'wait_interval': '30',
    'dataset_bucket' : 'gs://kfp-exp/fairseq-lm-data'
        }


@dsl.pipeline(
        name='KFP-Pipelines Example',
        description='Kubeflow pipeline using pre-built components'
        )
def pipeline(
    project_id=PROJECT_ID,
    region=REGION,
    args=common_args,
    master_image_uri=FAIRSEQ_IMAGE,
    job_id_prefix='',
    job_id='',
    wait_interval='30',
    dataset_bucket='gs://kfp-exp/fairseq-lm-data'
):
    """ Pipeline (DAG) definition """

    preprocess = preprocess_op (
            dataset_bucket=dataset_bucket,
            args_in=common_args
            )
    preprocess.execution_options.caching_strategy.max_cache_staleness = "P0D"

client = kfp.Client(host='321ff3bfe3fa6d70-dot-us-central2.pipelines.'
                    'googleusercontent.com')
client.create_run_from_pipeline_func(pipeline, pipeline_args)
