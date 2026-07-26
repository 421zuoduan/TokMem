from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from compositional_toolathlon.dataset import create_step_dataloader
from compositional_toolathlon.training import train_stepwise_model


class _TinyToolModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.trainable_logits = torch.nn.Parameter(torch.zeros(1, 3, 8))
        self.lora_config = None
        self.tool_reserved_token_ids = [5]
        self.token_id_to_tool_id = {5: 0}

    def get_trainable_parameters(self):
        return [self.trainable_logits]

    def forward(
        self,
        input_ids,
        attention_mask,
        *,
        return_hidden_states,
    ):
        del attention_mask
        if return_hidden_states is not True:
            raise AssertionError("training must request hidden states")
        logits = self.trainable_logits.expand(input_ids.shape[0], -1, -1)
        hidden_states = torch.zeros(
            input_ids.shape[0],
            input_ids.shape[1],
            4,
            device=input_ids.device,
        )
        return logits, hidden_states


class _IndexedStepDataset(torch.utils.data.Dataset):
    def __init__(self, size: int) -> None:
        self.data = [
            {"episode_id": f"episode-{index // 3}", "sample_id": f"step-{index}"}
            for index in range(size)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        return {
            "input_ids": torch.tensor([1, 2], dtype=torch.long),
            "attention_mask": torch.ones(2, dtype=torch.long),
            "labels": torch.tensor([-100, 2], dtype=torch.long),
            "available_tool_mask": torch.tensor([True]),
            "episode_weight": torch.tensor(1.0),
            "raw_data": self.data[index],
        }


class NoValidationTrainingTests(unittest.TestCase):
    def test_shuffled_step_loader_visits_every_step_once_per_epoch(self):
        dataset = _IndexedStepDataset(142)
        tokenizer = SimpleNamespace(pad_token_id=0)

        def two_epoch_orders():
            with patch(
                "compositional_toolathlon.dataset.ToolathlonStepDataset",
                return_value=dataset,
            ):
                loader = create_step_dataloader(
                    "unused.jsonl",
                    "unused-manifest.json",
                    tokenizer,
                    object(),
                    batch_size=4,
                    max_length=128,
                    use_eoc=False,
                    shuffle=True,
                    balance_by_episode=False,
                    sampler_seed=17,
                )
            return [
                [
                    record["sample_id"]
                    for batch in loader
                    for record in batch["raw_data"]
                ]
                for _ in range(2)
            ]

        first_run = two_epoch_orders()
        second_run = two_epoch_orders()
        self.assertEqual(first_run, second_run)
        self.assertNotEqual(first_run[0], first_run[1])
        for epoch_order in first_run:
            self.assertEqual(len(epoch_order), 142)
            self.assertEqual(len(set(epoch_order)), 142)

    def test_final_epoch_is_selected(self):
        model = _TinyToolModel()
        batch = {
            "input_ids": torch.tensor([[1, 5, 2]], dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "labels": torch.tensor([[-100, 5, 2]], dtype=torch.long),
            "available_tool_mask": torch.tensor([[True]]),
            "episode_weight": torch.tensor([1.0]),
        }
        metrics = train_stepwise_model(
            model=model,
            dataloader=[batch],
            num_epochs=2,
            lr=1e-3,
            device="cpu",
            use_logit_bias=False,
            use_logit_train_add=False,
        )
        self.assertEqual(len(metrics["epochs"]), 2)
        self.assertTrue(
            all("validation" not in epoch for epoch in metrics["epochs"])
        )
        self.assertEqual(
            metrics["checkpoint_selection"],
            {
                "metric": "final_epoch",
                "best_epoch": 2,
                "best_value": None,
            },
        )


if __name__ == "__main__":
    unittest.main()
