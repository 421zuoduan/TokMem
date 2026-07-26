from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

from compositional_intercode_bash.checkpoint import (
    base_model_load_location,
    build_base_model_identity,
    validate_base_model_identity,
)
from compositional_intercode_bash.memory_model import (
    MemoryForwardOutput,
    MemoryTokenRegistry,
    ProceduralMemoryModel,
    orthogonal_memory_rows,
)
from compositional_intercode_bash.training_data import (
    RoutingSites,
    add_train_routing_bias,
    gather_routing_sites,
    serialize_view,
    strip_control_token_ids,
)
from compositional_intercode_bash.tests.helpers import ByteTokenizer, TinyCausalLM


class MemoryModelTest(unittest.TestCase):
    def test_formal_base_model_identity_requires_commit_or_hashed_directory(self):
        with self.assertRaisesRegex(RuntimeError, "40-character commit"):
            build_base_model_identity(
                "organization/model",
                None,
                require_reproducible=True,
            )
        with self.assertRaisesRegex(RuntimeError, "40-character commit"):
            build_base_model_identity(
                "organization/model",
                "main",
                require_reproducible=True,
            )
        commit = "a" * 40
        remote = build_base_model_identity(
            "organization/model",
            commit,
            require_reproducible=True,
        )
        self.assertEqual(
            base_model_load_location(remote),
            ("organization/model", commit),
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("{}\n", encoding="utf-8")
            local = build_base_model_identity(
                str(root),
                None,
                require_reproducible=True,
            )
            self.assertEqual(base_model_load_location(local), (str(root), None))
            (root / "config.json").write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                validate_base_model_identity(local)

    def test_model_revision_is_forwarded_to_backbone_loader(self):
        tokenizer = ByteTokenizer(native_reserved=5)
        calls = []

        class FakeAutoModel:
            @classmethod
            def from_pretrained(cls, model_name, **kwargs):
                calls.append((model_name, kwargs))
                return TinyCausalLM(len(tokenizer), hidden_size=8)

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoModelForCausalLM = FakeAutoModel
        with mock.patch.dict(sys.modules, {"transformers": fake_transformers}):
            model = ProceduralMemoryModel.from_pretrained(
                "base-model",
                tokenizer,
                3,
                method="tokmem",
                initialization_seed=1,
                revision="pinned-revision",
                device="cpu",
                dtype=torch.float32,
            )
        self.assertEqual(calls[0][0], "base-model")
        self.assertEqual(calls[0][1]["revision"], "pinned-revision")
        model.close()

    def test_registry_expands_without_capping_k(self):
        tokenizer = ByteTokenizer(native_reserved=3)
        registry = MemoryTokenRegistry.build(tokenizer, 5)
        self.assertEqual(registry.num_procedures, 5)
        self.assertEqual(len(registry.procedure_token_ids), 5)
        self.assertEqual(registry.native_procedure_count, 2)
        self.assertEqual(len(registry.added_token_names), 3)
        self.assertNotIn(registry.eoc_token_id, registry.procedure_token_ids)
        for name, token_id in zip(
            registry.procedure_token_names,
            registry.procedure_token_ids,
        ):
            self.assertEqual(tokenizer.encode(name, add_special_tokens=False), [token_id])

    def test_all_rows_are_orthogonal_and_capacity_is_explicit(self):
        first = orthogonal_memory_rows(5, 8, seed=42)
        second = orthogonal_memory_rows(5, 8, seed=42)
        torch.testing.assert_close(first, second)
        normalized = torch.nn.functional.normalize(first.float(), dim=1)
        torch.testing.assert_close(
            normalized @ normalized.T,
            torch.eye(5),
            atol=1e-5,
            rtol=1e-5,
        )
        orthogonal_memory_rows(8, 8, seed=1)
        with self.assertRaisesRegex(ValueError, "K=9.*hidden_size=8"):
            orthogonal_memory_rows(9, 8, seed=1)

    def test_input_and_output_rows_are_overridden(self):
        tokenizer = ByteTokenizer(native_reserved=3)
        registry = MemoryTokenRegistry.build(tokenizer, 5)
        base = TinyCausalLM(len(tokenizer), hidden_size=8)
        model = ProceduralMemoryModel(
            base,
            registry,
            method="tokmem",
            initialization_seed=7,
        )
        token_id = registry.procedure_token_ids[-1]
        input_ids = torch.tensor([[token_id, tokenizer.byte_offset + ord("x")]])
        embedded = model._input_embeddings(input_ids)
        torch.testing.assert_close(embedded[0, 0], model.procedure_embeddings[-1])
        output = model(input_ids, torch.ones_like(input_ids))
        expected = output.final_hidden_state @ model.procedure_embeddings.T
        actual = output.logits[..., list(registry.procedure_token_ids)]
        torch.testing.assert_close(actual, expected)
        self.assertTrue(all(not parameter.requires_grad for parameter in base.parameters()))

    def test_targets_and_causal_route_sites(self):
        tokenizer = ByteTokenizer(native_reserved=5)
        registry = MemoryTokenRegistry.build(tokenizer, 3)
        view = {
            "presentation_id": "p0",
            "instruction": "do it",
            "command_raw": "  ab | c  # tail",
            "base_chunks": ["  ab | ", "c  # tail"],
            "segments": [
                {"piece_id": 0, "start": 0, "end": 1},
                {"piece_id": 2, "start": 1, "end": 2},
            ],
        }
        tokmem = serialize_view(
            tokenizer,
            registry,
            view,
            method="tokmem",
            max_length=10000,
            explicit_response_end_ids=[tokenizer.eot_id, tokenizer.eos_token_id],
        )
        tapmem = serialize_view(
            tokenizer,
            registry,
            view,
            method="tapmem",
            max_length=10000,
            explicit_response_end_ids=[tokenizer.eot_id, tokenizer.eos_token_id],
        )
        self.assertEqual(tokmem.memory_token_count, 2)
        self.assertEqual(tokmem.eoc_count, 0)
        self.assertEqual(tapmem.eoc_count, 2)
        self.assertEqual(tapmem.route_site_count, 2)

        prefix_ids = tokenizer.encode("  ", add_special_tokens=False)
        gap_ids = tokenizer.encode(" | ", add_special_tokens=False)
        suffix_ids = tokenizer.encode("  # tail", add_special_tokens=False)
        expected_tokmem_target = [
            *prefix_ids,
            registry.procedure_token_ids[0],
            *tokenizer.encode("ab", add_special_tokens=False),
            *gap_ids,
            registry.procedure_token_ids[2],
            *tokenizer.encode("c", add_special_tokens=False),
            *suffix_ids,
            tokenizer.eot_id,
            tokenizer.eos_token_id,
        ]
        expected_tapmem_target = [
            *prefix_ids,
            registry.procedure_token_ids[0],
            *tokenizer.encode("ab", add_special_tokens=False),
            registry.eoc_token_id,
            *gap_ids,
            registry.procedure_token_ids[2],
            *tokenizer.encode("c", add_special_tokens=False),
            registry.eoc_token_id,
            *suffix_ids,
            tokenizer.eot_id,
            tokenizer.eos_token_id,
        ]
        self.assertEqual(
            tokmem.input_ids[tokmem.prompt_length :],
            expected_tokmem_target,
        )
        self.assertEqual(
            tapmem.input_ids[tapmem.prompt_length :],
            expected_tapmem_target,
        )

        labels = torch.tensor([tapmem.labels])
        sites = gather_routing_sites(labels, registry)
        self.assertEqual(
            sites.time_indices.tolist()[0],
            tapmem.prompt_length + len(prefix_ids) - 1,
        )
        self.assertEqual(sites.targets.tolist(), [0, 2])
        self.assertEqual(sites.count, 2)
        self.assertEqual(
            tapmem.labels[sites.time_indices.tolist()[1]],
            gap_ids[-1],
        )

    def test_route_sites_allow_ordinary_prefix_and_delayed_memory(self):
        tokenizer = ByteTokenizer(native_reserved=5)
        registry = MemoryTokenRegistry.build(tokenizer, 3)
        ordinary_a = tokenizer.byte_offset + ord("a")
        ordinary_b = tokenizer.byte_offset + ord("b")
        labels = torch.tensor(
            [
                [
                    -100,
                    -100,
                    ordinary_a,
                    registry.procedure_token_ids[1],
                    ordinary_b,
                    registry.eoc_token_id,
                    ordinary_a,
                    ordinary_b,
                    registry.procedure_token_ids[2],
                ]
            ]
        )
        sites = gather_routing_sites(labels, registry)
        self.assertEqual(sites.time_indices.tolist(), [2, 7])
        self.assertEqual(sites.targets.tolist(), [1, 2])

    def test_multitoken_terminator_and_exact_control_removal(self):
        ordinary, missing = strip_control_token_ids(
            [100, 10, 90, 11, 102, 101, 12, 90, 91, 13, 100],
            procedure_token_ids=[100, 101],
            eoc_token_id=102,
            response_end_sequences=[[90, 91]],
        )
        self.assertEqual(ordinary, [10, 90, 11, 12])
        self.assertFalse(missing)
        ordinary, missing = strip_control_token_ids(
            [100, 10, 102, 11, 90],
            procedure_token_ids=[100, 101],
            eoc_token_id=102,
            response_end_sequences=[[90, 91]],
        )
        self.assertEqual(ordinary, [10, 11, 90])
        self.assertTrue(missing)

    def test_tcra_bias_uses_normalized_prior_only_at_route_sites(self):
        tokenizer = ByteTokenizer(native_reserved=5)
        registry = MemoryTokenRegistry.build(tokenizer, 3)
        base = TinyCausalLM(len(tokenizer), hidden_size=8)
        model = ProceduralMemoryModel(
            base,
            registry,
            method="tapmem",
            initialization_seed=3,
            logit_bias_scale=2.0,
        )
        with torch.no_grad():
            model.routing_head.weight.zero_()
            model.routing_head.bias.copy_(torch.tensor([0.0, 1.0, -1.0]))
        logits = torch.zeros(1, 4, len(tokenizer))
        boundary_hidden = torch.randn(2, 8)
        sites = RoutingSites(
            batch_indices=torch.tensor([0, 0]),
            time_indices=torch.tensor([1, 3]),
            targets=torch.tensor([0, 2]),
        )
        output = add_train_routing_bias(
            model,
            logits,
            boundary_hidden,
            sites,
        )
        expected = (
            torch.log_softmax(torch.tensor([0.0, 1.0, -1.0]), dim=-1)
            + torch.log(torch.tensor(3.0))
        ) * 2.0
        for time_index in (1, 3):
            torch.testing.assert_close(
                output[0, time_index, list(registry.procedure_token_ids)],
                expected,
            )
        self.assertEqual(
            torch.count_nonzero(output[0, [0, 2]]).item(),
            0,
        )
        ordinary_ids = [
            token_id
            for token_id in range(len(tokenizer))
            if token_id not in registry.procedure_token_ids
        ]
        self.assertEqual(
            torch.count_nonzero(output[..., ordinary_ids]).item(),
            0,
        )

    def test_tcra_detaches_boundary_hidden_but_trains_head(self):
        tokenizer = ByteTokenizer(native_reserved=5)
        registry = MemoryTokenRegistry.build(tokenizer, 3)
        model = ProceduralMemoryModel(
            TinyCausalLM(len(tokenizer), hidden_size=8),
            registry,
            method="tapmem",
            initialization_seed=4,
        )
        logits = torch.zeros(1, 1, len(tokenizer))
        boundary_hidden = torch.randn(1, 8, requires_grad=True)
        sites = RoutingSites(
            batch_indices=torch.tensor([0]),
            time_indices=torch.tensor([0]),
            targets=torch.tensor([0]),
        )
        output = add_train_routing_bias(
            model,
            logits,
            boundary_hidden,
            sites,
            detach_hidden=True,
        )
        output[0, 0, registry.procedure_token_ids[0]].backward()
        self.assertIsNone(boundary_hidden.grad)
        self.assertIsNotNone(model.routing_head.weight.grad)
        self.assertGreater(
            float(model.routing_head.weight.grad.abs().sum().item()),
            0.0,
        )

    def test_generation_bias_follows_delayed_raw_memory_candidates(self):
        tokenizer = ByteTokenizer(native_reserved=5)
        registry = MemoryTokenRegistry.build(tokenizer, 3)
        model = ProceduralMemoryModel(
            TinyCausalLM(len(tokenizer), hidden_size=8),
            registry,
            method="tapmem",
            initialization_seed=5,
        )
        ordinary_a = tokenizer.byte_offset + ord("a")
        ordinary_b = tokenizer.byte_offset + ord("b")
        terminator = [
            tokenizer.byte_offset + ord("x"),
            tokenizer.byte_offset + ord("y"),
        ]
        scripted_tokens = [
            ordinary_a,
            registry.eoc_token_id,
            ordinary_b,
            registry.procedure_token_ids[0],
            ordinary_a,
            registry.eoc_token_id,
            ordinary_b,
            registry.procedure_token_ids[2],
            ordinary_a,
            registry.eoc_token_id,
            *terminator,
        ]
        forward_calls = []
        bias_calls = []

        def scripted_forward(
            self,
            input_ids,
            attention_mask=None,
            *,
            past_key_values=None,
            use_cache=False,
            **kwargs,
        ):
            call_index = len(forward_calls)
            forward_calls.append(input_ids.detach().clone())
            logits = torch.zeros(
                input_ids.shape[0],
                input_ids.shape[1],
                len(tokenizer),
            )
            logits[:, -1, scripted_tokens[call_index]] = 10.0
            hidden = torch.zeros(
                input_ids.shape[0],
                input_ids.shape[1],
                self.hidden_size,
            )
            return MemoryForwardOutput(
                logits=logits,
                final_hidden_state=hidden,
                past_key_values=("cache", call_index),
                base_output=None,
            )

        def record_bias(
            self,
            full_logits,
            boundary_hidden,
            *,
            detach_hidden=True,
        ):
            bias_calls.append(len(forward_calls))
            return full_logits

        model.forward = types.MethodType(scripted_forward, model)
        model.add_routing_bias = types.MethodType(record_bias, model)
        prompt = torch.tensor([[ordinary_a]])
        result = model.generate_tokens(
            prompt,
            torch.ones_like(prompt),
            response_end_sequences=[terminator],
            max_new_tokens=len(scripted_tokens) + 2,
        )
        self.assertEqual(result["generated_ids"], scripted_tokens)
        self.assertTrue(result["terminated"])
        self.assertFalse(result["missing_terminator"])
        # The orphan EOC and ordinary gap do not trigger routing.  Each raw
        # memory candidate does, including one delayed after EOC + ordinary.
        self.assertEqual(bias_calls, [4, 8])
        self.assertEqual(result["memory_bank_constraint_trigger_count"], 2)
        self.assertEqual(result["memory_bank_constraint_changed_token_count"], 0)
        # The first terminator token alone did not stop generation.
        self.assertEqual(len(forward_calls), len(scripted_tokens))

    def test_tcra_rescore_can_leave_memory_bank_without_entering_procedure(self):
        tokenizer = ByteTokenizer(native_reserved=5)
        registry = MemoryTokenRegistry.build(tokenizer, 3)
        model = ProceduralMemoryModel(
            TinyCausalLM(len(tokenizer), hidden_size=8),
            registry,
            method="tapmem",
            initialization_seed=6,
        )
        ordinary = tokenizer.byte_offset + ord("a")
        terminator = [tokenizer.byte_offset + ord("x")]
        raw_tokens = [
            registry.procedure_token_ids[0],
            registry.procedure_token_ids[2],
            registry.eoc_token_id,
            *terminator,
        ]
        forward_calls = []
        bias_calls = []

        def scripted_forward(
            self,
            input_ids,
            attention_mask=None,
            *,
            past_key_values=None,
            use_cache=False,
            **kwargs,
        ):
            call_index = len(forward_calls)
            forward_calls.append(input_ids.detach().clone())
            logits = torch.zeros(
                input_ids.shape[0],
                input_ids.shape[1],
                len(tokenizer),
            )
            # A procedure token is the raw argmax, but the combined bank mass
            # remains below 0.5 because the rest of the vocabulary stays at 0.
            logits[:, -1, raw_tokens[call_index]] = 1.0
            hidden = torch.zeros(
                input_ids.shape[0],
                input_ids.shape[1],
                self.hidden_size,
            )
            return MemoryForwardOutput(
                logits=logits,
                final_hidden_state=hidden,
                past_key_values=("cache", call_index),
                base_output=None,
            )

        def rescore(
            self,
            full_logits,
            boundary_hidden,
            *,
            detach_hidden=True,
        ):
            bias_calls.append(len(forward_calls))
            if len(bias_calls) == 1:
                full_logits = full_logits.clone()
                full_logits[:, ordinary] = 20.0
            return full_logits

        model.forward = types.MethodType(scripted_forward, model)
        model.add_routing_bias = types.MethodType(rescore, model)
        prompt = torch.tensor([[ordinary]])
        result = model.generate_tokens(
            prompt,
            torch.ones_like(prompt),
            response_end_sequences=[terminator],
            max_new_tokens=8,
        )
        self.assertEqual(
            result["generated_ids"],
            [ordinary, registry.procedure_token_ids[2], registry.eoc_token_id, *terminator],
        )
        # The first rescore chose an ordinary token, so the model stayed
        # OUTSIDE and the next raw memory candidate was routed again.
        self.assertEqual(bias_calls, [1, 2])

    def test_memory_bank_probability_gate_is_tapmem_only_and_outside_only(self):
        tokenizer = ByteTokenizer(native_reserved=5)
        registry = MemoryTokenRegistry.build(tokenizer, 3)
        ordinary = tokenizer.byte_offset + ord("a")
        terminator = [tokenizer.byte_offset + ord("x")]

        def run(method):
            model = ProceduralMemoryModel(
                TinyCausalLM(len(tokenizer), hidden_size=8),
                registry,
                method=method,
                initialization_seed=7,
                memory_bank_probability_threshold=0.5,
            )
            forward_calls = []
            bias_calls = []

            def scripted_forward(
                self,
                input_ids,
                attention_mask=None,
                *,
                past_key_values=None,
                use_cache=False,
                **kwargs,
            ):
                call_index = len(forward_calls)
                forward_calls.append(input_ids.detach().clone())
                logits = torch.full(
                    (
                        input_ids.shape[0],
                        input_ids.shape[1],
                        len(tokenizer),
                    ),
                    -10.0,
                )
                # The ordinary token is the raw top-1, while three memory
                # tokens together carry more than half of the softmax mass.
                for token_id in registry.procedure_token_ids:
                    logits[:, -1, token_id] = 0.0
                logits[:, -1, ordinary if call_index == 0 else terminator[0]] = 0.2
                hidden = torch.zeros(
                    input_ids.shape[0],
                    input_ids.shape[1],
                    self.hidden_size,
                )
                return MemoryForwardOutput(
                    logits=logits,
                    final_hidden_state=hidden,
                    past_key_values=("cache", call_index),
                    base_output=None,
                )

            def record_bias(
                self,
                full_logits,
                boundary_hidden,
                *,
                detach_hidden=True,
            ):
                bias_calls.append(len(forward_calls))
                return full_logits

            model.forward = types.MethodType(scripted_forward, model)
            model.add_routing_bias = types.MethodType(record_bias, model)
            prompt = torch.tensor([[ordinary]])
            return model.generate_tokens(
                prompt,
                torch.ones_like(prompt),
                response_end_sequences=[terminator],
                max_new_tokens=3,
            ), bias_calls

        tapmem, tapmem_bias_calls = run("tapmem")
        self.assertEqual(
            tapmem["generated_ids"],
            [registry.procedure_token_ids[0], *terminator],
        )
        self.assertEqual(tapmem_bias_calls, [1])
        self.assertEqual(tapmem["memory_bank_constraint_trigger_count"], 1)
        self.assertEqual(
            tapmem["memory_bank_constraint_changed_token_count"],
            1,
        )
        # After the selected memory token, TapMem is INSIDE.  The second
        # distribution also has bank mass above 0.5, but remains unrestricted
        # and therefore selects the ordinary terminator.

        tokmem, tokmem_bias_calls = run("tokmem")
        self.assertEqual(tokmem["generated_ids"], [ordinary, *terminator])
        self.assertEqual(tokmem_bias_calls, [])
        self.assertEqual(tokmem["memory_bank_constraint_trigger_count"], 0)
        self.assertEqual(
            tokmem["memory_bank_constraint_changed_token_count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()
