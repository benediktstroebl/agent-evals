# import boto3
import glob
import json
import argparse
from openai import OpenAI
from tqdm import tqdm
# from botocore.config import Config

def read_txt_file(file_path):
    """
    Reads a text file and returns the content as a string.

    :param file_path: Path to the .txt file to be read.
    :return: Content of the file as a string.
    """
    try:
        with open(file_path, 'r') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        return "The file was not found."
    except Exception as e:
        return f"An error occurred: {e}"

def read_json_file(file_path):
    """
    Reads a JSON file and returns the content as a dictionary.

    :param file_path: Path to the .json file to be read.
    :return: Content of the file as a dictionary.
    """
    import json

    try:  
        with open(file_path, 'r') as file:
            content = json.load(file)
        return content
    except FileNotFoundError:
        return "The file was not found."
    except json.JSONDecodeError:
        return "The file is not a valid JSON file."
    except Exception as e:
        return f"An error occurred: {e}"
    
    
    
def truncate_and_combine(prefix, suffix, book, qas, n,  max_len):
    book_words = book.split(' ')
    max_len = min(max_len, len(book_words))
    book_words = book_words[:max_len - n*1000]
    book = ' '.join(book_words) + ' Book ends. '
    return prefix + book + suffix + qas

def get_gpt4_completion(prefix, suffix, book, qas, key,max_len=128000,  max_new_tokens=1000):
    client = OpenAI(api_key=key)
    n = 0
    while(n<100):
        try:
            prompt = truncate_and_combine(prefix, suffix, book, qas, n, max_len=max_len)
            completion = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=[{"role": "user", "content": prompt}],
                temperature = 0,
                top_p = 1,
                presence_penalty=0.,
                frequency_penalty=0.,
                max_tokens=max_new_tokens,
            )
            ans = completion.choices[0].message.content
            return ans
        except Exception as e:
            print(n, type(e), e)
            n+=1
    #     return "None"
    return "No Generation"


def make_data(book_data_path, qa_data_path, qa_type='multichoice', max_len = 128000):
    book_files = sorted(glob.glob(book_data_path+'/*'))
    qa_files = sorted(glob.glob(qa_data_path+'/*'))

    book_titles = [file.split('/')[-1][:-4].strip().lower() for file in book_files]
    qa_titles = [file.split('/')[-1][:-5].strip().lower() for file in qa_files]
    for i, title in enumerate(book_titles):
        if title != qa_titles[i]:
            raise Exception('The book files are inconsistent with the QA files; {}, {}'.format(title, qa_titles[i]))

    def read_txt(file_path):
        f = open(file_path, 'r', encoding='utf8')
        return f.read()
    def read_json(file_path):
        f = open(file_path, 'r', encoding='utf8')
        return json.load(f)

    def process_qas2multichoice(qa_path):
        qas = read_json(qa_path)
        tmp = 'Question starts: '
        # option_format = {'0':'A', '1':'B', '2':'C', '3':'D'}
        option_format = {0:'A', 1:'B', 2:'C', 3:'D'}
        for i, qa in enumerate(qas):
            tmp += 'Question {}:'.format(str(i))
            tmp += qa['Question']
            tmp += ' Options: '
            for j, c in enumerate(qa['Options']):
                tmp += ' {}:'.format(option_format[j])
                tmp += c
            tmp += '\n'
        return tmp
    def process_qas2generative(qa_path):
        qas = read_json(qa_path)
        tmp = 'Question starts: '
        for i, qa in enumerate(qas):
            tmp += 'Question {}:'.format(str(i))
            tmp += qa['Question']
            tmp += '\n'
        return tmp

    def process_book(book_path, title, max_len):
        context = read_txt(book_path)
        book = 'Book title: ' + title + ';'
        book += 'Book context: ' + context + ';'
        book_list = book.split(' ')[:int(max_len/1.3)]
        book = ' '.join(book_list)
        return book

    if qa_type == 'multichoice':
        eval_data = {title:{'book':process_book(book_files[i], title, max_len), 'qas':process_qas2multichoice(qa_files[i])} for i, title in enumerate(book_titles)}
    elif qa_type == 'generative':
        eval_data = {title:{'book':process_book(book_files[i], title, max_len), 'qas':process_qas2generative(qa_files[i])} for i, title in enumerate(book_titles)}
    elif qa_type == 'closebook-mc':
        eval_data = {title:{'book':'', 'qas':process_qas2multichoice(qa_files[i])} for i, title in enumerate(book_titles)}
    elif qa_type == 'closebook-gen':
        eval_data = {title:{'book':'', 'qas':process_qas2generative(qa_files[i])} for i, title in enumerate(book_titles)}
    return eval_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--book_data_path', type=str, required=True)
    parser.add_argument('--qa_data_path', type=str, required=True)
    parser.add_argument('--key_path', type=str, default=None)
    parser.add_argument('--model', type=str, choices=['gpt4', 'claude21', 'claude3'], default='gpt4')
    parser.add_argument('--output_path', type=str, default=None)
    parser.add_argument('--qa_type', type=str, choices=['multichoice', 'generative', 'closebook-mc', 'closebook-gen'], default='multichoice')
    parser.add_argument('--with_evidences', action="store_true")

    args = parser.parse_args()

    key = ...

    if args.model == 'claude21' or args.model == 'claude3':
        pass
        # config = Config(read_timeout=1000)
        # bedrock = boto3.client(service_name='bedrock-runtime', region_name='us-west-2', config=config)
    if args.qa_type == 'multichoice':
        prefix = "You are a literature professor. I will provide you with the full text of a novel along with a series of questions and corresponding choices pertaining to it. Please thoroughly analyze the novel 's content to accurately respond to each of the following questions.\n"
        if args.with_evidences == True:
            suffix = 'Try your best to answer the questions based on the given full text the novel. Your output format should be \'Answer0: <choice>; <evidences>\nAnswer1: <choice>; <evidences>\n...\nAnswern: <choice>; <evidences>\', each answer in one line and repeat the original text to support your answer.\n'
        else:
            suffix = 'Try your best to answer the questions based on the given full text the novel. Your output format should be \'Answer0: <choice>\nAnswer1: <choice>\n...\nAnswern: <choice>\'(only the choice index is required), each answer in one line without outputing the questions and other info.\n'
    elif args.qa_type == 'generative':
        prefix = "You are a literature professor. I will provide you with the full text of a novel along with a series of questions. Please thoroughly analyze the novel 's content to accurately respond to each of the following questions.\n"
        if args.with_evidences == True:
            suffix = 'Try your best to answer the questions based on the given full text the novel. The answer should be in short with only one or several words. Your output format should be \'Answer0: <answer>$ <evidences>\nAnswer1: <answer>$ <evidences>\n...\nAnswern: <answer>$ <evidences>\', each answer in one line with all the supporting evidences. Each evidence should be a sentence exactly from the original text without any paraphrase.\n'
        else:
            suffix = 'Try your best to answer the questions based on the given full text the novel. The answer should be in short with only one or several words. Your output format should be \'Answer0: <answer>\nAnswer1: <answer>\n...\nAnswern: <answer>\', each answer in one line without outputing the questions and other info.\n'
    elif args.qa_type == 'closebook-mc':
        prefix = "You are a literature professor. I will provide you a series of questions along with four choices for each question. Please accurately select the correct choice to each of the following questions.\n"
        suffix = 'Try your best to answer the questions based on your own knowledge. our output format should be \'Answer0: <choice>\nAnswer1: <choice>\n...\nAnswern: <choice>\' (only the choice index is required), each answer in one line without outputing the questions and other info.\n'
    elif args.qa_type == 'closebook-gen':
        prefix = "You are a literature professor. I will provide you a series of questions. Please accurately respond to each of the following questions.\n"
        suffix = 'Try your best to answer the questions based on your own knowledge. The answer should be in short with only one or several words. Your output format should be \'Answer0: <answer>\nAnswer1: <answer>\n...\nAnswern: <answer>\', each answer in one line without outputing the questions and other info.\n'

    if args.model == 'gpt4':
        max_len = 128000
    elif  args.model == 'claude21'or args.model == 'claude3':
        max_len = 200000
    else:
        max_len = 200000
    max_new_tokens = 4096 if args.qa_type == 'generative' else 768
    eval_data = make_data(args.book_data_path, args.qa_data_path, args.qa_type, max_len)
    book_titles = eval_data.keys()

    results = {}
    for title in tqdm(book_titles):
        print('On book: {}'.format(title))
        if args.model == 'gpt4':
            result = get_gpt4_completion(prefix, suffix, eval_data[title]['book'], eval_data[title]['qas'], key=key,  max_len=max_len, max_new_tokens=max_new_tokens)
        results[title] = result
        print(result)

    if args.output_path is not None:
        f = open(args.output_path, 'w', encoding='utf8')
        json.dump(results, f, indent=1)