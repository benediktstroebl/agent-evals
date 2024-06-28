import os
import logging
from logging_utils import JsonFormatter
import pandas as pd
import time
import re
import argparse
import itertools
from tqdm import tqdm

import os
import sys
import argparse
from typing import List
import shutil

import openai
from browser_env import ScriptBrowserEnv

from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent)+"/src")

from webagents_step.utils.data_prep import *
from webagents_step.agents.prompt_agent import PromptAgent
from webagents_step.agents.step_agent import StepAgent
from webagents_step.prompts.webarena import flat_fewshot_template, step_fewshot_template
from webagents_step.environment.webarena import WebArenaEnvironmentWrapper

openai.api_key = os.environ.get("OPENAI_API_KEY")

def run():
    parser = argparse.ArgumentParser(
        description="Only the config file argument should be passed"
    )
    parser.add_argument("--run_name", type=str, required=True, help="The name of the run")
    parser.add_argument("--config", type=str, required=True, help="yaml config file location")
    args = parser.parse_args()
    with open(args.config, "r") as file:
        config = DotDict(yaml.safe_load(file))
    
    dstdir = f"{config.logdir}/{args.run_name}/f{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(dstdir, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(dstdir, args.config.split("/")[-1]))
    random.seed(42)
    
    config_file_list = []
    
    # ids covered in gitlab
    task_ids = config.env.task_ids

    for task_id in task_ids:
        config_file_list.append(f"tasks/webarena/{task_id}.json")

    action_to_prompt_dict = {k: v for k, v in step_fewshot_template.__dict__.items() if isinstance(v, dict)}

    low_level_action_list = config.agent.low_level_action_list

    # Set up logger
    logger = logging.getLogger(args.run_name)
    logger.setLevel(logging.DEBUG)
    # Create a file handler to output logs to a file
    file_handler = logging.FileHandler(os.path.join(config.logdir, f'{args.run_name}', f'webarena_{config.agent.model_name}_{args.run_name}.log'))
    file_handler.setLevel(logging.DEBUG)
    # Set the JSON formatter for the handler
    file_handler.setFormatter(JsonFormatter())
    # Add the handler to the logger
    logger.addHandler(file_handler)

    logger.info("Starting the run", extra={"run_parameters": config.to_dict(), "type": "run_started"})

    if config.agent.type == "step":
        agent_init = lambda: StepAgent(
        root_action = config.agent.root_action,
        action_to_prompt_dict = action_to_prompt_dict,
        low_level_action_list = low_level_action_list,
        max_actions=config.env.max_env_steps,
        verbose=config.verbose,
        logging=config.logging,
        debug=config.debug,
        model=config.agent.model_name,
        prompt_mode=config.agent.prompt_mode,
        logger=logger
        )
    elif config.agent.type == "flat_fewshot8k":
        agent_init = lambda: PromptAgent(
            prompt_template=flat_fewshot_template.flat_fewshot_agent8k,
            model=config.agent.model_name,
            prompt_mode=config.agent.prompt_mode,
            max_actions=config.env.max_env_steps,
            verbose=config.verbose,
            logging=config.logging,
            debug=config.debug,
            logger=logger
        )
    elif config.agent.type == "flat_fewshot4k":
        agent_init = lambda: PromptAgent(
            prompt_template=flat_fewshot_template.flat_fewshot_agent4k,
            model=config.agent.model_name,
            prompt_mode=config.agent.prompt_mode,
            max_actions=config.env.max_env_steps,
            verbose=config.verbose,
            logging=config.logging,
            debug=config.debug,
            logger=logger
        )
    else:
        raise NotImplementedError(f"{config.agent.type} not implemented")

    #####
    # Evaluate
    #####
    # task_ids_affected_by_reddit_rate_limit = [600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 618, 619, 620, 621, 622, 623, 624, 626, 629, 630, 631, 632, 633, 634, 635, 636, 637, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649]
    task_ids_affected_by_reddit_rate_limit = []
    for i, config_file in enumerate(config_file_list):
        
        env = WebArenaEnvironmentWrapper(config_file=config_file, 
                                         max_browser_rows=config.env.max_browser_rows, 
                                         max_steps=config.env.max_env_steps, 
                                         slow_mo=1, 
                                         observation_type="accessibility_tree", 
                                         current_viewport_only=True, 
                                         viewport_size={"width": 1920, "height": 1080}, 
                                         headless=config.env.headless,
                                         logger=logger,
                                         model_name=config.agent.model_name)
        with open(config_file, "r") as f:
                task_config = json.load(f)
        if task_config["task_id"] in task_ids_affected_by_reddit_rate_limit:
            print("Sleeping for 21 mins") 
            # print current time
            print("Current time:", time.strftime('%Y%m%d-%H%M%S'))
            # sleep for 21 mins
            time.sleep(1260)
        logger.info(f"Starting {i+1}th task", extra={"task_id": task_config["task_id"], "type": "task_started"})
        start_time = time.time()
        agent = agent_init()
        objective = env.get_objective()
        try:
            status = agent.act(objective=objective, env=env)
        except Exception as e:
            # catch error and save latest status as task result
            print(f"Error in task {task_config['task_id']}", "Error:", e)
            logger.info(f"Error in task {task_config['task_id']}", extra={"task_id": task_config["task_id"], "type": "task_error"})
            status = env.status()
            print("Status:", status)
        end_time = time.time()
        env.close()

        if config.logging:
            logger.info(f"Time taken for {i+1}th task: {end_time - start_time}", extra={"task_time": end_time - start_time, 
                                                                                        "task_id": task_config["task_id"],
                                                                                        "type": "task_finished"})
            with open(config_file, "r") as f:
                task_config = json.load(f)
            log_file = os.path.join(dstdir, f"{task_config['task_id']}.json")
            log_data = {
                "task": config_file,
                "id": task_config['task_id'],
                "model": config.agent.model_name,
                "type": config.agent.type,
                "trajectory": agent.get_trajectory(),
            }
            summary_file = os.path.join(dstdir, "summary.csv")
            summary_data = {
                "task": config_file,
                "task_id": task_config['task_id'],
                "model": config.agent.model_name,
                "type": config.agent.type,
                "logfile": re.search(r"/([^/]+/[^/]+\.json)$", log_file).group(1),
            }
            summary_data.update(status)
            log_run(
                log_file=log_file,
                log_data=log_data,
                summary_file=summary_file,
                summary_data=summary_data,
            )
    
    # read in summary.csv and get accuracy
    summary_df = pd.read_csv(summary_file)
    success_rate = summary_df["success"].mean()
    logger.info("Finished run", extra={'accuracy': success_rate, "type": "run_finished"}) 
    
if __name__ == "__main__":
    run()
