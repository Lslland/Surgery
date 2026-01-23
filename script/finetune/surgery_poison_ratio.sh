#!/bin/bash

# ---------- env ----------
source ~/miniforge3/etc/profile.d/conda.sh
conda activate surgery

# ---------- Config ----------
POISON_RATIOS=(0 0.05 0.1 0.15 0.2)

device=${1:-0,1,2}

sample_num=1000
model_path=/models/Meta-Llama-3-8B-Instruct
benign_dataset=data/gsm8k.json
epochs=5
alpha=300
sink_token_position=0

cd ../../
PROJECT_ROOT=$(pwd)
echo "Current Project Root: $PROJECT_ROOT"

# ---------- Loop Start ----------
for poison_ratio in "${POISON_RATIOS[@]}"; do

    echo "=========================================================="
    echo ">>>> Starting Loop for Poison Ratio: $poison_ratio"
    echo "=========================================================="

    path_after_slash=$(basename "$model_path")
    
    #  PROJECT_ROOT
    output_dir=ckpt/gsm8k/${path_after_slash}_sink_f_${poison_ratio}_${sample_num}
    poison_eval_out=data/poison/gsm8k/${path_after_slash}_sink_f_${poison_ratio}_${sample_num}
    acc_eval_out=data/gsm8k/${path_after_slash}_sink_f_${poison_ratio}_${sample_num}

    mkdir -p "$(dirname "$output_dir")"
    mkdir -p "$(dirname "$poison_eval_out")"
    mkdir -p "$(dirname "$acc_eval_out")"

    echo "======================= RUN CONFIG ======================="
    echo "[Poison Ratio]          : $poison_ratio"
    echo "[Sample Number]         : $sample_num"
    echo "[Optimizer]             : surgery_sft"
    echo "[Alpha]                 : $alpha"
    echo "[Output Directory]      : $output_dir"
    echo "=========================================================="

    # ---------- Train ----------
    echo "--- Starting Sink-SFT Training ---"
    CUDA_VISIBLE_DEVICES=${device} python train.py \
      --model_name_or_path "${model_path}" \
      --data_path PKU-Alignment/BeaverTails_dangerous \
      --bf16 True \
      --output_dir "${output_dir}" \
      --num_train_epochs ${epochs} \
      --per_device_train_batch_size 10 \
      --per_device_eval_batch_size 10 \
      --gradient_accumulation_steps 1 \
      --save_strategy "steps" \
      --save_steps 100000 \
      --save_total_limit 0 \
      --learning_rate 5e-6 \
      --weight_decay 0 \
      --warmup_ratio 0.1 \
      --lr_scheduler_type "constant" \
      --logging_steps 10 \
      --tf32 True \
      --eval_steps 5000 \
      --cache_dir cache \
      --optimizer surgery_sft \
      --evaluation_strategy "steps" \
      --sample_num "${sample_num}" \
      --poison_ratio "${poison_ratio}" \
      --label_smoothing_factor 0 \
      --sink_token_position ${sink_token_position} \
      --benign_dataset "${benign_dataset}" \
      --alpha ${alpha}

    # ---------- Poison Evaluation ----------
    echo "--- Starting Poison Evaluation ---"
    cd poison/evaluation
    
    CUDA_VISIBLE_DEVICES=${device} python pred.py \
      --base_model_folder "${model_path}" \
      --model_folder ../../${output_dir}  \
      --output_path ../../${poison_eval_out}

    CUDA_VISIBLE_DEVICES=${device} python eval_sentiment.py \
      --input_path ../../${poison_eval_out}
    
    cd "$PROJECT_ROOT"

    # ---------- Accuracy Evaluation ----------
    echo "--- Starting Accuracy Evaluation ---"
    cd gsm8k
    
    CUDA_VISIBLE_DEVICES=${device} python pred_eval.py \
      --base_model_folder "${model_path}" \
      --model_folder ../${output_dir} \
      --output_path ../${acc_eval_out}
      
    cd "$PROJECT_ROOT"

done

echo "All runs completed!"
