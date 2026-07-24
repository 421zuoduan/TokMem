"""Resolve TokMem wrapper classes without changing existing backbone behavior."""

from transformers import AutoConfig

from model import FunctionCallingModel
from qwen35_model import Qwen35FunctionCallingModel


BACKBONE_MODEL_CLASSES = {
    "qwen3_5": Qwen35FunctionCallingModel,
}


def resolve_function_calling_model_class(model_name, local_files_only=True):
    """Return the wrapper class registered for a Hugging Face model type."""
    config = AutoConfig.from_pretrained(
        model_name,
        local_files_only=local_files_only,
    )
    return BACKBONE_MODEL_CLASSES.get(
        config.model_type,
        FunctionCallingModel,
    )
