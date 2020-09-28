#!/bin/bash
# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Submit CAIP (Hyperparameter tuning) Job

export IMAGE_URI=gcr.io/pytorch-tpu-nfs/fairseq-lm

now=$(date +"%Y%m%d_%H%M%S")

JOB_NAME="fairseq_hptune_$now"

REGION=us-central1

  gcloud ai-platform jobs submit training "$JOB_NAME" \
    --region "$REGION" \
    --master-image-uri "$IMAGE_URI" \
    --config config.yaml \
    -- \
    --task language_modeling   data-bin/wikitext-103   --save-dir checkpoints/transformer_wikitext-103   --arch transformer_lm --share-decoder-input-output-embed   --dropout 0.1   --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0   --lr-scheduler inverse_sqrt --warmup-updates 4000 --warmup-init-lr 1e-07   --tokens-per-sample 512 --sample-break-mode none   --max-tokens 1024 --update-freq 16   --fp16   --max-update 500
