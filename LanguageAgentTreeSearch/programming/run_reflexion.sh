run_names=("run1" "run2" "run3" "run4" "run5")
for run_name in "${run_names[@]}"; do
  python main.py \
    --run_name "test_react3" \
    --root_dir "root" \
    --dataset_path ./benchmarks/humaneval-py.jsonl \
    --strategy "reflexion" \
    --language "py" \
    --model "gpt-3.5-turbo" \
    --pass_at_k "1" \
    --max_iters "8" \
    --verbose
