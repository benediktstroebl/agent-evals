dataset=$1
fallback_name=$2
strategy="simple_boosting"
run_names=("run1" "run2" "run3" "run4" "run5")
for run_name in "${run_names[@]}"; do
  python main.py \
    --run_name $run_name/ \
    --root_dir ../output_data/$strategy/$dataset/$fallback_name/ \
    --dataset_path ../input_data/$dataset/dataset/probs.jsonl \
    --strategy $strategy \
    --model $run_name \
    --n_proc "1" \
    --testfile ../input_data/$dataset/test/tests.jsonl \
    --verbose \
    --port "8000"

done