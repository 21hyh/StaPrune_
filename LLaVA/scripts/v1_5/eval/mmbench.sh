#!/bin/bash
# FasterVLM Token Pruning Parameters
export TOK_KEEP_RATIO=0.111
export SELECTION_STRATEGY=auto
export HF_ENDPOINT=https://hf-mirror.com

export CUDA_VISIBLE_DEVICES=0,1,2,3
SPLIT="mmbench_dev_20230712"

python -m llava.eval.model_vqa_mmbench \
    --model-path /mnt/sda/*/workspace/intern/*/models/llava-v1.6-vicuna-7b-hf \
    --question-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --answers-file ./playground/data/eval/mmbench/answers/$SPLIT/llava-v1.5-7b.jsonl \
    --single-pred-prompt \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --tok-keep-ratio $TOK_KEEP_RATIO \
    --prune-method prunemerge \
    --selection-strategy $SELECTION_STRATEGY

mkdir -p playground/data/eval/mmbench/answers_upload/$SPLIT

python scripts/convert_mmbench_for_submission.py \
    --annotation-file ./playground/data/eval/mmbench/$SPLIT.tsv \
    --result-dir ./playground/data/eval/mmbench/answers/$SPLIT \
    --upload-dir ./playground/data/eval/mmbench/answers_upload/$SPLIT \
    --experiment llava-v1.5-7b
