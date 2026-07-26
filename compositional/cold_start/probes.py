"""Encode tool documentation into final hidden-state representations."""

import torch
import torch.nn.functional as functional

from backbone_prompting import format_user_assistant_prompt


DOCUMENT_TEMPLATE = """Read the following tool documentation.

{schema}
"""


def schema_text(schema):
    lines = [
        f"Tool name: {schema['name']}",
        f"Purpose: {schema.get('description', '')}",
        "Parameters:",
    ]
    parameters = schema.get("parameters", {})
    if not parameters:
        lines.append("- none")
    for name, specification in parameters.items():
        parameter_type = specification.get("type", "unknown")
        description = specification.get("description", "")
        lines.append(f"- {name} ({parameter_type}): {description}")
    return "\n".join(lines)


def purpose_schema(schema):
    """Keep the routing-relevant purpose while removing schema-format noise."""
    return {
        "name": schema["name"],
        "description": schema.get("description", ""),
        "parameters": {},
    }


def _right_pad(sequences, pad_token_id, device):
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), max_length),
        int(pad_token_id),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(sequences), max_length),
        dtype=torch.long,
        device=device,
    )
    lengths = []
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row, :length] = 1
        lengths.append(length)
    return input_ids, attention_mask, lengths


def _final_hidden_states_without_logits(model, input_ids, attention_mask):
    """Run the backbone only; the vocabulary projection is not needed here."""
    causal_lm = model._get_core_model()
    backbone = getattr(causal_lm, causal_lm.base_model_prefix)
    outputs = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    return outputs.last_hidden_state


def extract_document_hidden_states(
    model,
    tokenizer,
    schemas,
    batch_size=8,
    document_view="full",
):
    """Return one normalized final hidden state for every tool document."""
    sequences = []
    for schema in schemas:
        if document_view == "purpose":
            schema = purpose_schema(schema)
        elif document_view != "full":
            raise ValueError(f"Unknown document view: {document_view}")
        document = DOCUMENT_TEMPLATE.format(schema=schema_text(schema))
        prompt = format_user_assistant_prompt(
            tokenizer,
            document,
            model=model,
        )
        sequences.append(
            tokenizer(prompt, add_special_tokens=False)["input_ids"]
        )

    vectors = []
    device = model.trainable_tool_input_embeddings.device
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        input_ids, attention_mask, lengths = _right_pad(
            batch,
            tokenizer.pad_token_id,
            device,
        )
        with torch.inference_mode():
            hidden_states = _final_hidden_states_without_logits(
                model,
                input_ids,
                attention_mask,
            )
        row_indices = torch.arange(len(batch), device=device)
        end_indices = torch.tensor(lengths, device=device) - 1
        selected = hidden_states[row_indices, end_indices].float()
        vectors.append(
            functional.normalize(selected, p=2, dim=-1).cpu()
        )
    return torch.cat(vectors, dim=0)
