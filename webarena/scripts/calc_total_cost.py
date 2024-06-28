import json
import argparse

def calculate_tokens(log_file_path):
  """
  Calculates the total number of prompt_tokens and completion_tokens from a log file.

  Args:
    log_file_path: Path to the log file.

  Returns:
    A tuple containing the total prompt_tokens and completion_tokens.
  """

  total_prompt_tokens = 0
  total_completion_tokens = 0

  with open(log_file_path, 'r') as log_file:
    for line in log_file:
      try:
        event = json.loads(line.strip())
      except json.JSONDecodeError:
        print(f"Skipping invalid line: {line}")
        continue

      if 'prompt_tokens' in event and 'completion_tokens' in event:
        total_prompt_tokens += event['prompt_tokens']
        total_completion_tokens += event['completion_tokens']

  return total_prompt_tokens, total_completion_tokens

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Calculate prompt and completion tokens from a log file.")
  parser.add_argument("--log_file", help="Path to the log file")
  parser.add_argument("--prompt_tokens_price", help="Price per 1M tokens", type=float)
  parser.add_argument("--completion_tokens_price", help="Price per 1M tokens", type=float)
  args = parser.parse_args()

  prompt_tokens, completion_tokens = calculate_tokens(args.log_file)

  print(f"Total prompt tokens: {prompt_tokens}")
  print(f"Total completion tokens: {completion_tokens}")
  print(f"Total cost: ${prompt_tokens * (args.prompt_tokens_price/1000000) + completion_tokens * (args.completion_tokens_price/1000000)}")
