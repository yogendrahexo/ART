from unsloth.models import FastLanguageModel  # type: ignore
from peft.peft_model import PeftModel
from transformers import PreTrainedTokenizerBase

from ..config.model import ModelConfig


def get_model_and_tokenizer(
    config: ModelConfig,
) -> tuple[PeftModel, PreTrainedTokenizerBase]:
    model, tokenizer = FastLanguageModel.from_pretrained(**config.init_args)
    model = FastLanguageModel.get_peft_model(**config.peft_args)

    return model, tokenizer
