import json

def get_reddit_task_ids(json_file_path):
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        
    reddit_task_ids = [task['task_id'] for task in data if 'reddit' in task['sites']]
    
    return reddit_task_ids

# Example usage
json_file_path = 'config_files/test.raw.json'
reddit_task_ids = get_reddit_task_ids(json_file_path)
print("Nr of Reddit tasks: ", len(reddit_task_ids))
print(reddit_task_ids)
