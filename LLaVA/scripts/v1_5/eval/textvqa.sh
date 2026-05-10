#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAVA_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DEFAULT_MODEL_PATH="/mnt/sda/*/workspace/intern/*/models/llava-v1.5-7b"
MODEL_PATH="${1:-${MODEL_PATH:-$DEFAULT_MODEL_PATH}}"
CKPT_NAME="${CKPT_NAME:-$(basename "$MODEL_PATH")}"
ABLATION_MODE="${ABLATION_MODE:-full}"
RUN_TAG="${RUN_TAG:-}"

DEFAULT_DATA_ROOT="/mnt/sda/*/workspace/intern/*/FasterVLM_v99_test2/LLaVA/playground/data/eval/textvqa"
TEXTVQA_DATA_ROOT="${TEXTVQA_DATA_ROOT:-}"

has_textvqa_data() {
    local root="$1"
    [[ -f "$root/llava_textvqa_val_v051_ocr.jsonl" ]] && \
    [[ -f "$root/TextVQA_0.5.1_val.json" ]] && \
    [[ -d "$root/train_images" ]]
}

if [[ -z "$TEXTVQA_DATA_ROOT" ]]; then
    if has_textvqa_data "$LLAVA_ROOT/playground/data/eval/textvqa"; then
        TEXTVQA_DATA_ROOT="$LLAVA_ROOT/playground/data/eval/textvqa"
    else
        TEXTVQA_DATA_ROOT="$DEFAULT_DATA_ROOT"
    fi
fi

if [[ ! -f "$TEXTVQA_DATA_ROOT/llava_textvqa_val_v051_ocr.jsonl" ]]; then
    echo "TextVQA question file not found: $TEXTVQA_DATA_ROOT/llava_textvqa_val_v051_ocr.jsonl" >&2
    exit 1
fi

if [[ ! -f "$TEXTVQA_DATA_ROOT/TextVQA_0.5.1_val.json" ]]; then
    echo "TextVQA annotation file not found: $TEXTVQA_DATA_ROOT/TextVQA_0.5.1_val.json" >&2
    exit 1
fi

if [[ ! -d "$TEXTVQA_DATA_ROOT/train_images" ]]; then
    echo "TextVQA image folder not found: $TEXTVQA_DATA_ROOT/train_images" >&2
    exit 1
fi

OUTPUT_ROOT="$LLAVA_ROOT/playground/data/eval/textvqa/answers"
mkdir -p "$OUTPUT_ROOT"
FILE_STEM="$CKPT_NAME"
if [[ -n "$RUN_TAG" ]]; then
    FILE_STEM="${FILE_STEM}_${RUN_TAG}"
elif [[ "$ABLATION_MODE" != "full" ]]; then
    SAFE_MODE="${ABLATION_MODE//[^A-Za-z0-9._-]/_}"
    FILE_STEM="${FILE_STEM}_${SAFE_MODE}"
fi
ANSWERS_FILE="$OUTPUT_ROOT/${FILE_STEM}.jsonl"
CHUNK_ROOT="$OUTPUT_ROOT/${FILE_STEM}_chunks"
mkdir -p "$CHUNK_ROOT"

TOK_KEEP_RATIO="${TOK_KEEP_RATIO-0.333}"
PRUNE_METHOD="${PRUNE_METHOD-prunemerge}"
SELECTION_STRATEGY="${SELECTION_STRATEGY-auto}"
CONV_MODE="${CONV_MODE:-vicuna_v1}"
TEMPERATURE="${TEMPERATURE:-0}"
GPU_LIST="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$GPU_LIST"
CHUNKS=${#GPULIST[@]}

BASE_CMD=(
    python -m llava.eval.model_vqa_loader
    --model-path "$MODEL_PATH"
    --question-file "$TEXTVQA_DATA_ROOT/llava_textvqa_val_v051_ocr.jsonl"
    --image-folder "$TEXTVQA_DATA_ROOT/train_images"
    --temperature "$TEMPERATURE"
    --conv-mode "$CONV_MODE"
)

if [[ -n "$TOK_KEEP_RATIO" ]]; then
    BASE_CMD+=(--tok-keep-ratio "$TOK_KEEP_RATIO")

    if [[ -n "$PRUNE_METHOD" ]]; then
        BASE_CMD+=(--prune-method "$PRUNE_METHOD")
    fi

    if [[ -n "$SELECTION_STRATEGY" ]]; then
        BASE_CMD+=(--selection-strategy "$SELECTION_STRATEGY")
    fi
fi

cd "$LLAVA_ROOT"

echo "LLaVA root: $LLAVA_ROOT"
echo "Model path: $MODEL_PATH"
echo "TextVQA data root: $TEXTVQA_DATA_ROOT"
echo "Answers file: $ANSWERS_FILE"
echo "GPU list: $GPU_LIST"
echo "Chunks: $CHUNKS"
echo "Ablation mode: $ABLATION_MODE"

if [[ "$CHUNKS" -eq 1 ]]; then
    "${BASE_CMD[@]}" --answers-file "$ANSWERS_FILE" --ablation-mode "$ABLATION_MODE"
else
    for IDX in $(seq 0 $((CHUNKS - 1))); do
        CHUNK_FILE="$CHUNK_ROOT/${CHUNKS}_${IDX}.jsonl"
        rm -f "$CHUNK_FILE"

        CUDA_VISIBLE_DEVICES="${GPULIST[$IDX]}" \
        "${BASE_CMD[@]}" \
            --answers-file "$CHUNK_FILE" \
            --ablation-mode "$ABLATION_MODE" \
            --num-chunks "$CHUNKS" \
            --chunk-idx "$IDX" &
    done

    wait

    : > "$ANSWERS_FILE"
    for IDX in $(seq 0 $((CHUNKS - 1))); do
        cat "$CHUNK_ROOT/${CHUNKS}_${IDX}.jsonl" >> "$ANSWERS_FILE"
    done
fi

python -m llava.eval.eval_textvqa \
    --annotation-file "$TEXTVQA_DATA_ROOT/TextVQA_0.5.1_val.json" \
    --result-file "$ANSWERS_FILE"
