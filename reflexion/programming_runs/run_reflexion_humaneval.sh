model=$1

run_names=("run1" "run2" "run3" "run4" "run5")
for run_name in "${run_names[@]}"; do
  python main.py \
  --run_name $run_name \
  --root_dir ../output_data/reflexion/humaneval/$model/ \
  --dataset_path ./benchmarks/humaneval-py.jsonl \
  --strategy "reflexion" \
  --language "py" \
  --model $model \
  --pass_at_k "1" \
  --max_iters "2" \
  --verbose
done
