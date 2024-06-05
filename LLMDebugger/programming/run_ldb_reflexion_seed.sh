dataset=$1
model=$2
seedmodel=$3
strategy="ldb"
run_names=("run1" "run2" "run3" "run4" "run5")
for run_name in "${run_names[@]}"; do
  python main.py \
    --run_name $run_name/ \
    --root_dir ../output_data/$strategy/$dataset/$model+reflexion/ \
    --dataset_path ../input_data/$dataset/dataset/probs.jsonl \
    --strategy $strategy \
    --model $model \
    --seedfile /scratch/gpfs/bs6865/agent-eval/reflexion/output_data/reflexion/$dataset/$seedmodel/$run_name/humaneval-py._reflexion_2_${seedmodel}_pass_at_k_1_py.jsonl \
    --pass_at_k "1" \
    --max_iters "10" \
    --n_proc "1" \
    --port "8000" \
    --testfile ../input_data/$dataset/test/tests.jsonl \
    --verbose
done