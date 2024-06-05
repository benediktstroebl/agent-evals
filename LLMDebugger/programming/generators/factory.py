from .py_generate import PyGenerator
from .model import CodeLlama, ModelBase, GPT4, GPT35, StarCoder
import logging

def model_factory(model_name: str, port: str = "", key: str = "", logger: logging.Logger = None, client_type="azure") -> ModelBase:
    if "gpt-4" in model_name:
        return GPT4(model_name, key, logger=logger, client_type=client_type)
    elif "gpt-3.5" in model_name:
        return GPT35(model_name, key, logger=logger, client_type=client_type)
    elif model_name == "starcoder":
        return StarCoder(port, logger=logger)
    elif "llama" in model_name:
        if "codellama" in model_name:
            return CodeLlama("codellama/CodeLlama-34b-Instruct-hf", port, logger=logger)
        elif "llama3-8b" in model_name:
            return CodeLlama("meta-llama/Llama-3-8b-chat-hf", port, logger=logger)
        elif "llama3-70b" in model_name:
            return CodeLlama("meta-llama/Llama-3-70b-chat-hf", port, logger=logger)
        else:
            raise ValueError(f"Invalid model name: {model_name}")
    else:
        raise ValueError(f"Invalid model name: {model_name}")
