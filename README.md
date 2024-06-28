# Repository to preprint: AI agent evaluations that matter

This repository contains the accompanying code to the preprint with the title **AI agent evaluations that matter** by Sayash Kapoor, Benedikt Stroebl, Zachary S. Siegel, Nitya Nadgir, and Arvind Narayanan. 

Part of the analysis for this blog post builds on the following publications and their accompanying code repositories, which we used for reproducing their work.

#### HumanEval

**Reflexion ---**
[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) ([GitHub](https://github.com/noahshinn/reflexion/blob/main/programming_runs/simple.py)) (Copyright (c) 2023 Noah Shinn)

**LDB ---**
[LDB: A Large Language Model Debugger via Verifying Runtime Execution Step by Step](https://arxiv.org/abs/2402.16906) ([GitHub](https://github.com/floridsleeves/llmdebugger)) ([license](https://github.com/benediktstroebl/agent-evals/blob/9ea981d6656509e3632f1203e70cfac64730e15b/ldb_license.txt))

**LATS ---**
[Language Agent Tree Search Unifies Reasoning Acting and Planing in Language Models](https://arxiv.org/abs/2310.04406) ([GitHub](https://github.com/andyz245/LanguageAgentTreeSearch)) (Copyright (c) 2023 Andy Zhou)

#### WebArena

**WebArena ---**
[WebArena: A Realistic Web Environment for Building Autonomous Agents](https://arxiv.org/pdf/2307.13854.pdf) ([GitHub](https://github.com/web-arena-x/webarena)) ([license](https://github.com/benediktstroebl/agent-evals/blob/364c8c19c036a98d6f52203740afe9053fd88094/webarena_license.txt))

**STeP ---**
[SteP: Stacked LLM Policies for Web Actions](https://arxiv.org/abs/2310.03720) ([GitHub](https://github.com/asappresearch/webagents-step))

#### NovelQA

The [NovelQA website](https://github.com/NovelQA/novelqa.github.io) lists steps for downloading the data and evaluating the results.

The code for the evaluations presented in NovelQA is not yet available online. Authors of NovelQA shared their evaluation code with permission to share publicly; we use this for our analysis.

### General notes

#### Structure

- This repository is organized so that each high-level agent has its own dedicated directory, mirroring the structure of the original repositories associated with the respective research papers.
- Additionally, our baseline agents on HumanEval are built upon the LDB codebase and are therefore housed within the LLMDebugger directory.
- Each of the three primary directories of the HumanEval agents includes an `output_data/` folder, which contains the result logs from the experiments we conducted. For STeP, the log files and full agent traces are contained in the `data/` directory. For WebArena, we uploaded the traces and renderings for both agent specifications to [Dropbox](https://www.dropbox.com/scl/fo/h8ju8wgaa8bljk8ma1leb/AA3i9HHFaHvymKt-F_u10ug?rlkey=l0m8669krshy6um0agpl9jlxt&st=yxcg94ob&dl=0).
    
#### Logging on HumanEval 

- To track inference times and costs associated with the agents, we added code at relevant points within the source code. The resulting log files are stored alongside the results from solving the HumanEval tasks in the `output_data/` subdirectories located within each agent directory.
- **Note on interrupted runs:** Some experimental runs were interrupted mid way through the HumanEval problems (e.g., due to budget limits or network errors). We restarted these runs from the point of interruption (i.e., starting at the earliest unsolved HumanEval problem) to conserve costs. This means that the accuracy reflected in the LATS `.jsonl` files is not accurate in these instances ([example](https://github.com/benediktstroebl/agent-eval/blob/2a5afc1a29e539b28a870b7431d9b9a3bc4f21ef/LanguageAgentTreeSearch/output_data/lats/humaneval/gpt-4-turbo-2024-04-09/run1/humaneval-py._mcts_8_gpt-4-turbo-2024-04-09_pass_at_k_1_py.jsonl)). You can refer to the respective `.log` files stored in the same folder for the correct accuracy numbers. (To be clear, this only affects the log files stored by LATS. It does not affect the actual accuracy of the agent, nor the results reported in the blog post.)

#### Changes made to source code of agent papers

In order to reproduce the work of the publications mentioned above and to address encountered reproducibility issues, we had to make changes to the original code as provided by the authors. All of these changes are part of the commit history of this repository and can be inspected transparently. For more details on some of the reproducibility issues, please refer to the accompanying blog post.

#### Additional questions and details

You can refer to the blog post and its associated appendices for more details on our setup. You can contact us at [sayashk, stroebl, arvindn]@princeton.edu

## Example of Pareto frontier calculation

In our analysis, Pareto frontiers are employed to evaluate agent designs. We define the Pareto frontier as the set of points (agents) that are non-dominated by any other agent in terms of mean cost and accuracy. The frontier is constrained to be convex, meaning if two agents lie next to each other on the frontier, any linear combination of these agents should also yield a point that lies on the frontier curve. We provide a simple example implementation of how we calculate Pareto frontiers on simulated agent evaluation data: [Jupyter Notebook](https://github.com/benediktstroebl/agent-evals/blob/01241aaf0ebc9b5f418769afbf0289f0df3f3241/pareto_frontier_example.ipynb) and [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1Yxb8pwY_QhJd50GgICOmFbe2rFWCFHC5?usp=sharing)


## Running agents and models

### HumanEval

To set up the environments to run each agent, follow these steps:

1. Clone this repo and move to the respective agent directory:
```bash
git clone https://github.com/benediktstroebl/agent-eval.git
```

2. For each agent repository, create an environment with the provided module dependencies contained in the respective folder:
```bash
pip install -r requirements.txt
```

3. Set `OPENAI_API_KEY` environment variable to your OpenAI API key:
```bash
export OPENAI_API_KEY=<your key>
```

#### To run LDB agents, simple models, and baselines

##### To run simple models and baselines

- `Simple models` -  This uses the simple strategy implemented in the code accompanying the LDB agent for zero-shot evaluations of language models (i.e., there is no agent architecture).

    ```bash
    cd ./programming
    ./run_simple.sh humaneval [model] [output_dir]
    ```

 - `Escalation` - We modify the simple strategy of LDB but switch the underlying model to a more expensive one if a proposed solution fails at least one of the example tests. Running the script below will start five runs with `llama-3-8b-chat-hf`, `gpt-3.5-turbo-0125`, ​​`llama-3-70b-chat-hf`, `gpt-4-turbo-2024-04-09` as backend fallback models.

    ```bash
    cd ./programming
    ./run_simple_boosting.sh humaneval [name_you_can_set]
    ```

 - `Retry` - Simple strategy that repeatedly prompts the same language model, keeping all parameters equal across retrials, as long as the code outputted by the model failed at least one of the example tests.

    ```bash
    cd ./programming
    ./run_simple_repeat.sh humaneval [model] [name_you_can_set]
    ```

 - `Warming` - For the Warming baseline, we modify the Retry baseline by gradually increasing the temperature parameter across successive trials.

    ```bash
    cd ./programming
    ./run_simple_incr_temp.sh humaneval [model] [name_you_can_set]
    ```

#### To run LDB agents

 LDB agents require a seed file containing already-existing solutions from a model or agent, which the LDB agent then debugs. To start an LDB agent from scratch, first create the seed files using the steps listed in the simple agents and models part above.
 
 - `LDB with seed from simple strategy` - Use this if you want to reproduce LDB agents that use a seed generated using the simple models or agents. The resulting folder containing the outputs and logs will follow the nomenclature **model**+**seedmodel**.

    ```bash
    cd ./programming
    ./run_ldb.sh humaneval [model] [seedmodel]
    ```
    **Note:** This assumes that the respective seed is already in the `output_data/` directory at the appropriate location.

 - `LDB with Reflexion seed` - Use this if you want to reproduce LDB agents that use a seed generated with Reflexion. The resulting folder containing the outputs and logs will follow the nomenclature **model**+reflexion.

    ```bash
    cd ./programming
    ./run_ldb_reflexion_seed.sh humaneval [model] [seedmodel]
    ```
    **Note:** This assumes that the respective seed is already in the output_data directory at the appropriate location in the `reflexion/` directory.


#### To run LATS agents

 `LATS` - This reproduces our runs of the LATS agents. 

```bash
cd ./programming
./run_lats_humaneval.sh [model] [nr_int_tests]
```
**Note:** We learned from correspondence with the original authors, that the number of internal test cases was set to 6 for GPT-3.5 and 4 for GPT-4, respectively. The blog post has more details.

#### To run Reflexion agents

 `Reflexion` - This reproduces our runs of the Reflexion agents.

```bash
cd ./programming_runs
./run_reflexion_humaneval.sh [model]
```

### HotPotQA

To set up the environments to run our analysis, follow these steps:

1. Clone this repo and move to the `HotPotQA` directory:
```bash
git clone https://github.com/benediktstroebl/agent-eval.git
```

2. In the repository, create an environment with the provided module dependencies contained in the folder:
```bash
pip install -r requirements.txt
```
**Note:** This install a custom version of the `dspy` library that contains our joint optimizer for accuracy and cost. You can alternatively achieve the same thing by running the following command:
```bash
pip install git+https://github.com/benediktstroebl/dspy.git#egg=dspy_ai
```

#### To run baselines and agents on HotPotQA

For each of the four specifications in our HotPotQA analysis, the `retrieval_score` directory contains a separate directory. These directories contain the results of our analysis as well as detailed results of each inference that was done during evaluation.

In order to reproduce our analysis, including optimization and evaluation, you simply have to run the `.ipynb` notebooks for the respective model and agent. At the beginning of each notebook, you need to provide an API key (i.e., either OpenAI or Together.ai).

### NovelQA

#### To run GPT-4 on NovelQA

1. Download the NovelQA dataset from HuggingFace by following the steps outlined on the [NovelQA website](https://github.com/NovelQA/novelqa.github.io).
2. Specify your OpenAI API key in line 149.
3. Run the novelQA-gpt-4.py script after specifying the book data path and qa data path arguments, and leaving the rest of the arguments as the default.
4. For evaluation, upload the results to the Codabench evaluation website following the steps outlined in the [NovelQA website](https://github.com/NovelQA/novelqa.github.io).

#### To run our RAG agent on NovelQA
1. Download the NovelQA dataset from HuggingFace by following the steps outlined on the [NovelQA website](https://github.com/NovelQA/novelqa.github.io). Store it in ./NovelQA.
2. Specify your OpenAI API key.
3. Run each cell of the Jupyter Notebook in succession.
4. For evaluation, upload the results to the Codabench evaluation website following the steps outlined in the [NovelQA website](https://github.com/NovelQA/novelqa.github.io).

### WebArena

To set up the environments to run each agent, follow these steps:

1. Clone this repo and move to the respective agent directory:
```bash
git clone https://github.com/benediktstroebl/agent-eval.git
```

2. In the repository, create an environment with the provided module dependencies contained in the folder:
```bash
pip install -r requirements.txt
```

#### To run original WebArena baseline agents

To set up the environment and run the experiments, please refer to the instructions in the official [WebArena repository](https://github.com/web-arena-x/webarena?tab=readme-ov-file#end-to-end-evaluation). This repository assumes you have a working WebArena environment and completed all the steps outlined (except running the evaluation).

**Note:** You do not need to clone the WebArena repository. All the code required to setup the environment, including the steps outlined at the provided link to generate the test data and auto-log scripts. As pointed out in the paper, we had to modidfy some of the original code in order to reproduce the agents.

To run the evaluations:

- `GPT-4 (CoT, no UA Hint)` - This reproduces our runs of the GPT-4 (CoT, no UA Hint) agent.

    ```bash
    python run.py --instruction_path agent/prompts/jsons/p_cot_id_actree_2s_no_na.json --test_start_idx 0 --test_end_idx 812 --model gpt-4-turbo-2024-04-09 --result_dir results/cot_no_ua_hint/gpt-4-turbo-2024-04-09/run1
    ```

 - `GPT-3.5 (CoT, UA Hint)` - This reproduces our runs of the GPT-3.5 (CoT, UA Hint) agent.

    ```bash
    python run.py --instruction_path agent/prompts/jsons/p_cot_id_actree_2s.json --test_start_idx 0 --test_end_idx 812 --model gpt-3.5-turbo-0125 --result_dir results/cot_with_ua_hint/gpt-3.5-turbo-0125/run1
    ```

#### To run STeP agents

To set up the environment and run the experiments, please refer to the instructions in the official [WebArena repository](https://github.com/web-arena-x/webarena?tab=readme-ov-file#end-to-end-evaluation) as well as the [STeP repository](https://github.com/asappresearch/webagents-step). This repository assumes you have a working WebArena environment and completed all the steps outlined (except running the evaluation) as well as the few extra steps described in the STeP repository.

To run the evaluations:

- `STeP (GPT-4)` - This reproduces our runs of the STeP agent on the full WebArena benchmark.

    ```bash
    python scripts/evaluate/eval_webarena.py --config configs/webarena/eval_openai_agent_full.yml --run_name run1
    ```
    **Note:** The authors evaluate their agent on each WebArena site separately and modify the agent prompt based on the website of the given task. We follow this practice in our evaluations. We included the exact `task_ids` for each website as comments in the `eval_openai_agent_full.yml` file. In order to evaluate on the full benchmark, you need to run the agent on each site and uncomment the respective line in the file as well as set the corresponding `root_action` (as indicated with comment as well).

 - `STeP (GPT-4, Reddit only)` - This reproduces our runs of the STeP agent on all tasks that require the agent to interact with the Reddit site.

    ```bash
    python scripts/evaluate/eval_webarena.py --config configs/webarena/eval_openai_agent_reddit.yml --run_name run1
    ```
