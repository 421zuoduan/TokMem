import os

from accelerate import Accelerator, DataLoaderConfiguration, init_empty_weights
from accelerate.utils import FullyShardedDataParallelPlugin, broadcast_object_list
from transformers import AutoConfig, AutoModelForCausalLM


def infer_transformer_layer_cls_name(model_name):
    """Infer the decoder block class name for transformer-based FSDP auto-wrap."""
    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        backbone = AutoModelForCausalLM.from_config(config)

    layer_candidates = [
        getattr(getattr(backbone, "model", None), "layers", None),
        getattr(backbone, "layers", None),
        getattr(getattr(backbone, "transformer", None), "h", None),
    ]
    for layers in layer_candidates:
        if layers is not None and len(layers) > 0:
            return layers[0].__class__.__name__
    return None


def build_accelerator(args, model_name, use_seedable_sampler=False):
    """Build an Accelerator with optional FSDP wrapping."""
    dataloader_config = DataLoaderConfiguration(
        non_blocking=getattr(args, "pin_memory", False),
        use_seedable_sampler=use_seedable_sampler,
    )

    fsdp_plugin = None
    if getattr(args, "use_fsdp", False):
        layer_cls_name = infer_transformer_layer_cls_name(model_name)
        if layer_cls_name is None:
            raise ValueError(
                "--use_fsdp requires a recognized transformer decoder block class. "
                f"Failed to infer one for model {model_name!r}."
            )
        fsdp_plugin = FullyShardedDataParallelPlugin(
            sharding_strategy=args.fsdp_sharding_strategy,
            backward_prefetch=(
                None if args.fsdp_backward_prefetch == "none" else args.fsdp_backward_prefetch
            ),
            auto_wrap_policy="transformer_based_wrap",
            transformer_cls_names_to_wrap=[layer_cls_name],
            state_dict_type="FULL_STATE_DICT",
            use_orig_params=True,
            limit_all_gathers=True,
            cpu_ram_efficient_loading=True,
        )

    return Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        gradient_accumulation_steps=getattr(args, "gradient_accumulation_steps", 1),
        dataloader_config=dataloader_config,
        fsdp_plugin=fsdp_plugin,
    )


def configure_fsdp_ignored_modules(model, accelerator):
    """Keep small trainable helper modules replicated outside backbone FSDP sharding."""
    if accelerator is None or getattr(accelerator.distributed_type, "name", "") != "FSDP":
        return

    fsdp_plugin = getattr(accelerator.state, "fsdp_plugin", None)
    if fsdp_plugin is None:
        return
    if not hasattr(model, "get_fsdp_trainable_modules"):
        return

    ignored_modules = list(model.get_fsdp_trainable_modules())
    if not ignored_modules:
        return

    for module in ignored_modules:
        module.to(accelerator.device)
    fsdp_plugin.ignored_modules = ignored_modules


def synchronize_path(path_value, accelerator):
    """Broadcast a path from rank 0 so every rank follows the same load/save target."""
    if accelerator is None or accelerator.num_processes <= 1:
        return path_value

    payload = [path_value]
    broadcast_object_list(payload, from_process=0)
    return payload[0]


def is_rank_zero():
    return os.environ.get("RANK", "0") == "0"
