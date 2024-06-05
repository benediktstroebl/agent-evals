from .py_generate import PyGenerator
from .rs_generate import RsGenerator
from .go_generate import GoGenerator
from .generator_types import Generator
from .model import CodeLlama, ModelBase, GPT4, GPT35, StarChat, GPTDavinci
import logging


def generator_factory(lang: str) -> Generator:
    if lang == "py" or lang == "python":
        return PyGenerator()
    elif lang == "rs" or lang == "rust":
        return RsGenerator()
    elif lang == "go" or lang == "golang":
        return GoGenerator()
    else:
        raise ValueError(f"Invalid language for generator: {lang}")


def model_factory(model_name: str, logger: logging.Logger, client_type="azure") -> ModelBase:
    if "gpt-4" in model_name:
        return GPT4(model_name=model_name, logger=logger, client_type=client_type)
    elif "gpt-3.5" in model_name:
        return GPT35(model_name=model_name, logger=logger, client_type=client_type)
    elif model_name == "starchat":
        return StarChat()
    elif model_name.startswith("codellama"):
        # if it has `-` in the name, version was specified
        kwargs = {}
        if "-" in model_name:
            kwargs["version"] = model_name.split("-")[1]
        return CodeLlama(**kwargs)
    elif model_name.startswith("text-davinci"):
        return GPTDavinci(model_name)
    else:
        raise ValueError(f"Invalid model name: {model_name}")
