python main.py \
  --run_name "gpt4_mcts_3_full" \
  --root_dir "root" \
  --dataset_path ./benchmarks/humaneval-py.jsonl \
  --strategy "mcts" \
  --language "py" \
  --model "gpt-4-1106-preview" \
  --pass_at_k "1" \
  --max_iters "8" \
  --verbose
