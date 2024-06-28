import json
import argparse

def delete_unfinished_tasks(log_file_path):
  """
  Deletes blocks in the log file corresponding to unfinished tasks.

  Args:
    log_file_path: Path to the log file.
  """

  with open(log_file_path, 'r') as log_file:
    log_data = log_file.readlines()

  filtered_log_data = []
  task_log_data = []
  current_task_id = None
  task_started = False
  handled_task_ids = []

  for line in log_data:
    try:
      event = json.loads(line.strip())
    except json.JSONDecodeError:
      print(f"Skipping invalid line: {line}")
      continue

    if event['type'] == "run_started":
       filtered_log_data.insert(0, line)

    # if this is the last line in the log file, and the type is "run_finished", add it to the filtered log data
    if event['type'] == "run_finished" and line == log_data[-1]:
       filtered_log_data.append(line)
    
    if event['type'] == 'task_started':
      if task_started:
        print(f"Task started before the previous one finished. Previous task: {current_task_id}")
        task_log_data = []
        task_log_data.append(line)
        print(f"New task started: {event['task_id']}")
        current_task_id = event['task_id']
      else:
        print(f"Task started: {event['task_id']}")
        current_task_id = event['task_id']
        task_started = True
        task_log_data = []
        task_log_data.append(line)
    elif event['type'] == 'task_finished':
       print(f"Task finished: {event['task_id']}")
       if current_task_id == event['task_id']:
            if current_task_id in handled_task_ids:
                print(f"Task {current_task_id} already handled.")
                task_started = False
            else:
                handled_task_ids.append(current_task_id)
                task_started = False
                task_log_data.append(line)
                filtered_log_data.extend(task_log_data)
       else:
            print(f"Task finished before it started: {event['task_id']}")
            raise ValueError(f"Task finished before it started: {event['task_id']}")
    elif task_started:
      task_log_data.append(line)
    else:
      continue

  with open(log_file_path, 'w') as log_file:
    log_file.writelines(filtered_log_data)

if __name__ == "__main__":
    # parse log path from command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str)
    args = parser.parse_args()
    
    delete_unfinished_tasks(args.log_file)