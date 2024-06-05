import os
os.environ[
    "SHOPPING"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7770"
os.environ[
    "SHOPPING_ADMIN"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:7780/admin"
os.environ[
    "REDDIT"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:9999"
os.environ[
    "GITLAB"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8023/"
os.environ[
    "MAP"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:3000"
os.environ[
    "WIKIPEDIA"
] = "http://ec2-3-131-244-37.us-east-2.compute.amazonaws.com:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
os.environ[
    "HOMEPAGE"
] = "PASS"  # The home page is not currently hosted in the demo site


from webagents_step.environment.env import WebEnvironment
import json
import re
# Init an environment
from browser_env import (
    create_id_based_action,
    StateInfo,
    Trajectory,
    ActionTypes,
    ScriptBrowserEnv
)
from evaluation_harness.evaluators import evaluator_router

import traceback

class WebArenaEnvironmentWrapper(WebEnvironment):
    def __init__(self, config_file, max_browser_rows=300, max_steps=50, slow_mo=1, observation_type="accessibility_tree", current_viewport_only=False, viewport_size={"width": 1280, "height": 720}, headless=False, logger=None, model_name=None):
        self.webarena_env = ScriptBrowserEnv(
                    headless=headless,
                    slow_mo=slow_mo,
                    observation_type=observation_type,
                    current_viewport_only=current_viewport_only,
                    viewport_size=viewport_size
                )
        self.config_file = config_file
        with open(self.config_file, "r") as f:
            self.config = json.load(f)
        
        self.obs, self.info = self.webarena_env.reset(options={"config_file": self.config_file})
        self.terminated = False
        self.objective = self.config["intent"]
        self.url = self.config["start_url"]
        self.max_browser_rows = max_browser_rows
        self.max_steps = max_steps
        self.steps = 0
        self.is_done = False
        self.reward = 0.0

        self.logger = logger
        self.model_name = model_name
        
        self.trajectory: Trajectory = []
        self.update_webarena_metrics()
        
    def reset(self):
        self.obs, self.info = self.webarena_env.reset(options={"config_file": self.config_file})

    def close(self):
        self.webarena_env.close()
        
    def get_url(self):
        return self.url
    
    def get_objective(self):
        return self.objective 
        
    def observation(self): 
        self.obs = self.webarena_env._get_obs()
        self.url = self.webarena_env.page.url
        browser_content = self.obs["text"]
        browser_content = browser_content.split("\n")[:self.max_browser_rows] 
        browser_content = "\n".join(browser_content)
        return browser_content
    
    def done(self):
        if self.is_done:
            return True
        return False
    
    def status(self):
        return {'done': self.is_done, 'reward': self.reward, 'success': float(self.reward > 0), 'num_actions': self.steps}

    def step(self, action):
        self.steps = self.steps + 1

        
        if self.steps > self.max_steps:
            print(f"Steps {self.steps} exceeded maximum {self.max_steps}")
            self.is_done = True
            action_cmd = create_id_based_action("stop [N/A]")
            self.update_webarena_metrics(action_cmd)
            return self.status()

        if action is None or action is "" or ("note [" in action):
            print(f"[Step {self.steps}] {action}")
            action_cmd = None
        else:
            #### The blow two blocks are commented out to disable the rerouting of find_user policy reddit and the formatting fix of the action that we did for some experiments compared to original authors code
            # if "type" in action:
            #     # action got formatted wrong sometimes, so we need to reformat it
            #     parts = action.split()
            #     formatted_string = f"type [{parts[1]}] [{' '.join(parts[2:-1]).strip()}] [{parts[-1]}]"
            #     # replace "[[" and "]]" with single brackets
            #     action = formatted_string.replace("[[", "[").replace("]]", "]")
            #     # replace – with space
            #     action = action.replace("-", " ").replace("–", " ")

            ### REROUTING FIND_USER POLICY REDDIT (DECOMMENT TO ENABLE REROUTING)
            # if ("goto [" in action) and ("/user/" in action):
            #     print("REROUTING")
            #     action = action.replace("/user/", "/user_name/")
            # elif ("goto [" in action) and ("/user_name/" in action):
            #     print("REROUTING")
            #     action = action.replace("/user_name/", "/user/")

            print(f"[Step {self.steps}] {action}")
            action_cmd = create_id_based_action(action)

        if action_cmd:
            try:
                self.obs, _, self.terminated, _, self.info = self.webarena_env.step(action_cmd)
                self.update_webarena_metrics(action_cmd)
            except Exception as e:
                print(f"Error occurred while taking step: {e}")
            
        return self.status()
    
    def update_webarena_metrics(self, action_cmd=None):
        # Append action (if any) and resulting sate
        if action_cmd:
            self.trajectory.append(action_cmd)
            if action_cmd["action_type"]== ActionTypes.STOP:
                self.is_done = True

        if not self.is_done: # If we are done, no need to append state
            state_info: StateInfo = {"observation": self.obs, "info": self.info}
            self.trajectory.append(state_info)
            
        if self.is_done:    
            try:
                evaluator = evaluator_router(self.config_file, logger=self.logger, model_name=self.model_name)
                self.reward = evaluator(trajectory=self.trajectory, config_file=self.config_file, page=self.webarena_env.page, client=self.webarena_env.get_page_client(self.webarena_env.page))
            except Exception as e:
                print(f"Got excepetion: {e}")
                print(traceback.format_exc())
                self.reward = 0