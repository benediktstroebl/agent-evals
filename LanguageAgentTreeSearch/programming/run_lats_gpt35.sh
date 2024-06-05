python main.py \
  --run_name "gpt35_mcts" \
  --root_dir "root" \
  --dataset_path ./benchmarks/humaneval-py.jsonl \
  --strategy "mcts" \
  --language "py" \
  --model "gpt-3.5-turbo-0613" \
  --pass_at_k "1" \
  --max_iters "8" \
  --verbose
