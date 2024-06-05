from typing import List, Union, Optional, Literal
import dataclasses
import os
from vllm import LLM, SamplingParams
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
from openai import OpenAI
from transformers import GPT2Tokenizer, AutoTokenizer
import time
import os 
import logging
from openai import AzureOpenAI
from config import TOGETHER_API_KEY, HF_TOKEN
os.environ["TOGETHER_API_KEY"] = TOGETHER_API_KEY
os.environ["HF_TOKEN"] = HF_TOKEN

MessageRole = Literal["system", "user", "assistant"]

@dataclasses.dataclass()
class Message():
    role: MessageRole
    content: str


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message.content}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt_completion(
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    stop_strs: Optional[List[str]] = None,
    temperature: float = 0.0,
    num_comps=1,
) -> Union[List[str], str]:
    response = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
        n=num_comps,
    )
    if num_comps == 1:
        return response.choices[0].text  # type: ignore

    return [choice.text for choice in response.choices]  # type: ignore


def change_messages(tokenizer, messages, max_len):
    if isinstance(messages, str):
        message_lines = messages.split("\n")
        acc_msg_len = 0
        new_messages = ""
        for l in reversed(message_lines):
            acc_msg_len += len(tokenizer.tokenize(l))
            if acc_msg_len < max_len:
                new_messages = l + "\n" + new_messages
            else:
                break
        new_messages = new_messages.strip()
        return new_messages
    else:
        original_messages = messages
        new_messages = messages[:1]
        total_msg_len = len(tokenizer.tokenize(messages[0].content))
        rest_messages = []
        for msg in reversed(messages[1:]):
            msg_len = len(tokenizer.tokenize(msg.content))
            if msg_len + total_msg_len < max_len:
                rest_messages = [msg] + rest_messages
                total_msg_len += msg_len
            else:
                break
        messages = new_messages + rest_messages
    return messages

class ModelBase():
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        raise NotImplementedError


class GPTChat(ModelBase):
    def __init__(self, model_name: str, key: str = "", logger: logging.Logger = None, client_type: str = "azure"):
        self.name = model_name
        self.logger = logger
        self.is_chat = True
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if key != "":
            if client_type == "openai":
                self.client = OpenAI()
            else:
                if "gpt-4-turbo-0125" in model_name:
                    self.client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                    api_version="2023-12-01-preview",
                    azure_endpoint="https://gpt-35-turbo-agent-eval.openai.azure.com/"
                    )
                    self.name = "gpt-4-0125" # 0125
                elif "gpt-3.5-turbo-0125" in model_name:
                    self.client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                    api_version="2023-12-01-preview",
                    azure_endpoint="https://gpt-35-turbo-agent-eval.openai.azure.com/"
                )
                    self.name = "gpt-35-turbo"
                elif "gpt-4-0613" in model_name:
                    self.client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                    api_version="2023-12-01-preview",
                    azure_endpoint="https://agent-eval-platform.openai.azure.com/"
                )
                    self.name = "azure_gpt-4-0613"
                elif "gpt-3.5-turbo-0613" in model_name:
                    self.client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                    api_version="2023-12-01-preview",
                    azure_endpoint="https://agent-eval-platform.openai.azure.com/"
                )
                    self.name = "azure_gpt-35-turbo-0613"
                    
                else:
                    raise ValueError("Invalid model name for Azure backend!")
        else:
            if client_type == "openai":
                self.client = OpenAI()
            else:
                if "gpt-4-turbo-0125" in model_name:
                    self.client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                    api_version="2023-12-01-preview",
                    azure_endpoint="https://gpt-35-turbo-agent-eval.openai.azure.com/"
                    )
                    self.name = "gpt-4-0125" # 0125
                elif "gpt-3.5-turbo-0125" in model_name:
                    self.client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                    api_version="2023-12-01-preview",
                    azure_endpoint="https://gpt-35-turbo-agent-eval.openai.azure.com/"
                )
                    self.name = "gpt-35-turbo"
                elif "gpt-4-0613" in model_name:
                    self.client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                    api_version="2023-12-01-preview",
                    azure_endpoint="https://agent-eval-platform.openai.azure.com/"
                )
                    self.name = "azure_gpt-4-0613"
                elif "gpt-3.5-turbo-0613" in model_name:
                    self.client = AzureOpenAI(
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
                    api_version="2023-12-01-preview",
                    azure_endpoint="https://agent-eval-platform.openai.azure.com/"
                )
                    self.name = "azure_gpt-35-turbo-0613"
                    
                else:
                    raise ValueError("Invalid model name for Azure backend!")
    
    def gpt_chat(
        self,
        messages,
        stop: List[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        num_comps=1,
    ) -> Union[List[str], str]:
        try:
            new_messages = change_messages(self.tokenizer, messages, 3097)
            messages = new_messages
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.name,
                messages=[dataclasses.asdict(message) for message in messages],
                temperature=temperature,
                top_p=1,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                n=num_comps,
                stop=stop
            )
            end_time = time.time()

            self.logger.info("OpenAI API call", extra={"input_messages": [dataclasses.asdict(message) for message in messages],
                                                            "output_messages": response.choices[0].message.content if num_comps == 1 else [choice.message.content for choice in response.choices],
                                                           "prompt_tokens": response.usage.prompt_tokens,
                                                           "completion_tokens": response.usage.completion_tokens, 
                                                           "total_tokens": response.usage.total_tokens,
                                                            "inference_time": end_time - start_time,
                                                            "temperature": temperature,
                                                           "model_name": self.name,
                                                           "max_tokens": max_tokens,
                                                           "type": "api_call"})
        except Exception as e:
            print("GPT Error:", str(e))
            if "context_length_exceeded" in str(e):
                messages = change_messages(self.tokenizer, messages, 2097)
                print("AFTER CHANGE MESSAGE LEN:", len(messages))
                print(messages)
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[dataclasses.asdict(message) for message in messages],
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    n=num_comps,
                )
                end_time = time.time()
                self.logger.info("OpenAI API call", extra={"input_messages": [dataclasses.asdict(message) for message in messages],
                                                            "output_messages": response.choices[0].message.content if num_comps == 1 else [choice.message.content for choice in response.choices],
                                                           "prompt_tokens": response.usage.prompt_tokens,
                                                           "completion_tokens": response.usage.completion_tokens, 
                                                           "total_tokens": response.usage.total_tokens,
                                                           "temperature": temperature,
                                                            "inference_time": end_time - start_time,
                                                           "model_name": self.name,
                                                           "max_tokens": max_tokens,
                                                           "type": "api_call"})
            else:
                assert False, "GPT API error: " + str(e)
        if num_comps == 1:
            return response.choices[0].message.content  # type: ignore
        return [choice.message.content for choice in response.choices]  # type: ignore

    def generate_chat(self, messages: List[Message], stop: List[str] = None, max_tokens: int = 1024, temperature: float = 0.0, num_comps: int = 1) -> Union[List[str], str]:
        res = self.gpt_chat(messages, stop, max_tokens, temperature, num_comps)
        return res


class GPT4(GPTChat):
    def __init__(self, model_name, key, logger, client_type="azure"):
        super().__init__(model_name, key, logger, client_type=client_type)


class GPT35(GPTChat):
    def __init__(self, model_name, key, logger, client_type="azure"):
        super().__init__(model_name, key, logger, client_type=client_type)


class VLLMModelBase(ModelBase):
    """
    Base for huggingface chat models
    """

    def __init__(self, model, port="8000", logger: logging.Logger = None):
        super().__init__(model)
        self.model = model
        self.name = model
        self.logger = logger
        self.vllm_client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        self.together_client = OpenAI(api_key=os.environ.get("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1")

        if model == "meta-llama/Llama-3-70b-chat-hf":
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-70B-Instruct", token=HF_TOKEN)
        elif model == "meta-llama/Llama-3-8b-chat-hf":
            self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", token=HF_TOKEN)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_length = 7000
    
    def vllm_chat(
        self,
        prompt: str,
        stop: List[str] = [""],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        num_comps=1,
    ) -> Union[List[str], str]:
        max_length = self.max_length
        while True:
            prompt = change_messages(self.tokenizer, prompt, max_length)  # StarCoder max length
            try:
                start_time = time.time()
                responses = self.together_client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    echo=False,
                    max_tokens=max_tokens,
                    temperature=0,
                    top_p=1,
                    stop=stop,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    n=num_comps                
                    )
                end_time = time.time()
                self.logger.info("Together API call", extra={"input_messages": prompt,
                                                           "output_messages": responses.choices[0].text if num_comps == 1 else [response.choices[0].text for response in responses], 
                                                           "prompt_tokens": responses.usage.prompt_tokens,
                                                           "completion_tokens": responses.usage.completion_tokens, 
                                                           "total_tokens": responses.usage.total_tokens,
                                                           "temperature": temperature,
                                                            "inference_time": end_time - start_time,
                                                           "model_name": self.name,
                                                            "max_tokens": max_tokens,
                                                           "type": "api_call"})

            except Exception as e:
                print("VLLM Error:", str(e))
                if "maximum context length" in str(e):
                    max_length -= 2000
                else:
                    assert False, "VLLM API error: " + str(e)
            else:
                break
        if num_comps == 1:
            return responses.choices[0].text  # type: ignore
        return [response.choices[0].text for response in responses]  # type: ignore

    def generate_completion(self, messages: str, stop: List[str] = [""], max_tokens: int = 1024, temperature: float = 0.0, num_comps: int = 1) -> Union[List[str], str]:
        ret = self.vllm_chat(messages, stop, max_tokens, temperature, num_comps)
        return ret

    def prepare_prompt(self, messages: List[Message]):
        prompt = ""
        for i, message in enumerate(messages):
            prompt += message.content + "\n"
            if i == len(messages) - 1:
                prompt += "\n"
        return prompt

    def extract_output(self, output: str) -> str:
        return output


class StarCoder(VLLMModelBase):
    def __init__(self, port=""):
        super().__init__("bigcode/starcoder", port)


class CodeLlama(VLLMModelBase):
    def __init__(self, model_name, port="", logger: logging.Logger = None):
        super().__init__(model_name, port, logger)
