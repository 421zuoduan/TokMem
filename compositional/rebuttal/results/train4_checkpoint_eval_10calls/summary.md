# 4-call Checkpoints Evaluated on 10-call Test

- source summary: `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/summary.md`
- test split: `/data/shilong/tokmem/results/compositional/all_methods/data/test/function_calling_test_tools51-100_10calls.json`
- max new tokens: `512`
- eval batch size: `8`

## Aggregate

| Model | Method | Trials | Tool Sequence Exact | Tool Multiset Exact | Call Exact | Argument F1 | Tool F1 | Parse Error Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama1b | tokmem | 3 | 0.3208 | 0.3225 | 0.1683 | 0.6228 | 0.8192 | 0.2108 |
| llama1b | tapmem | 3 | 0.5746 | 0.5754 | 0.3029 | 0.7783 | 0.9429 | 0.0529 |
| llama3b | tokmem | 3 | 0.2512 | 0.2537 | 0.1325 | 0.5959 | 0.8117 | 0.3162 |
| llama3b | tapmem | 3 | 0.8600 | 0.8600 | 0.4350 | 0.8513 | 0.9873 | 0.0387 |
| llama8b | tokmem | 3 | 0.2988 | 0.3412 | 0.2313 | 0.7388 | 0.8756 | 0.1263 |
| llama8b | tapmem | 3 | 0.8500 | 0.8512 | 0.4163 | 0.8415 | 0.9855 | 0.0275 |

## Per Trial

### llama1b / tokmem

| Trial | Run | Samples | Tool Sequence Exact | Call Exact | Argument F1 | Tool F1 | Checkpoint |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | llama1b_tokmem_eoc_logit_bias_trial1_seed42 | 800 | 0.3325 | 0.1688 | 0.6369 | 0.8271 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama1b_tokmem_eoc_logit_bias_trial1_seed42/round_1_tools_51_100.pt` |
| 2 | llama1b_tokmem_eoc_logit_bias_trial2_seed42 | 800 | 0.3137 | 0.1575 | 0.5960 | 0.8086 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama1b_tokmem_eoc_logit_bias_trial2_seed42/round_1_tools_51_100.pt` |
| 3 | llama1b_tokmem_eoc_logit_bias_trial3_seed42 | 800 | 0.3162 | 0.1787 | 0.6353 | 0.8218 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama1b_tokmem_eoc_logit_bias_trial3_seed42/round_1_tools_51_100.pt` |

### llama1b / tapmem

| Trial | Run | Samples | Tool Sequence Exact | Call Exact | Argument F1 | Tool F1 | Checkpoint |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | llama1b_adap_tokmem_eoc_logit_bias_trial1_seed42 | 800 | 0.5500 | 0.2800 | 0.7690 | 0.9400 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama1b_adap_tokmem_eoc_logit_bias_trial1_seed42/round_2_tools_51_100.pt` |
| 2 | llama1b_adap_tokmem_eoc_logit_bias_trial2_seed42 | 800 | 0.5938 | 0.3225 | 0.7869 | 0.9464 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama1b_adap_tokmem_eoc_logit_bias_trial2_seed42/round_2_tools_51_100.pt` |
| 3 | llama1b_adap_tokmem_eoc_logit_bias_trial3_seed42 | 800 | 0.5800 | 0.3063 | 0.7791 | 0.9424 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama1b_adap_tokmem_eoc_logit_bias_trial3_seed42/round_2_tools_51_100.pt` |

### llama3b / tokmem

| Trial | Run | Samples | Tool Sequence Exact | Call Exact | Argument F1 | Tool F1 | Checkpoint |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | llama3b_tokmem_eoc_logit_bias_trial1_seed42 | 800 | 0.2512 | 0.1325 | 0.5959 | 0.8117 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama3b_tokmem_eoc_logit_bias_trial1_seed42/round_1_tools_51_100.pt` |
| 2 | llama3b_tokmem_eoc_logit_bias_trial2_seed42 | 800 | 0.2512 | 0.1325 | 0.5959 | 0.8117 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama3b_tokmem_eoc_logit_bias_trial2_seed42/round_1_tools_51_100.pt` |
| 3 | llama3b_tokmem_eoc_logit_bias_trial3_seed42 | 800 | 0.2512 | 0.1325 | 0.5959 | 0.8117 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama3b_tokmem_eoc_logit_bias_trial3_seed42/round_1_tools_51_100.pt` |

### llama3b / tapmem

| Trial | Run | Samples | Tool Sequence Exact | Call Exact | Argument F1 | Tool F1 | Checkpoint |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | llama3b_adap_tokmem_eoc_logit_bias_trial1_seed42 | 800 | 0.8600 | 0.4350 | 0.8513 | 0.9873 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama3b_adap_tokmem_eoc_logit_bias_trial1_seed42/round_2_tools_51_100.pt` |
| 2 | llama3b_adap_tokmem_eoc_logit_bias_trial2_seed42 | 800 | 0.8600 | 0.4350 | 0.8513 | 0.9873 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama3b_adap_tokmem_eoc_logit_bias_trial2_seed42/round_2_tools_51_100.pt` |
| 3 | llama3b_adap_tokmem_eoc_logit_bias_trial3_seed42 | 800 | 0.8600 | 0.4350 | 0.8513 | 0.9873 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama3b_adap_tokmem_eoc_logit_bias_trial3_seed42/round_2_tools_51_100.pt` |

### llama8b / tokmem

| Trial | Run | Samples | Tool Sequence Exact | Call Exact | Argument F1 | Tool F1 | Checkpoint |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | llama8b_tokmem_eoc_logit_bias_trial1_seed42 | 800 | 0.2988 | 0.2313 | 0.7388 | 0.8756 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama8b_tokmem_eoc_logit_bias_trial1_seed42/round_1_tools_51_100.pt` |
| 2 | llama8b_tokmem_eoc_logit_bias_trial2_seed42 | 800 | 0.2988 | 0.2313 | 0.7388 | 0.8756 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama8b_tokmem_eoc_logit_bias_trial2_seed42/round_1_tools_51_100.pt` |
| 3 | llama8b_tokmem_eoc_logit_bias_trial3_seed42 | 800 | 0.2988 | 0.2313 | 0.7388 | 0.8756 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama8b_tokmem_eoc_logit_bias_trial3_seed42/round_1_tools_51_100.pt` |

### llama8b / tapmem

| Trial | Run | Samples | Tool Sequence Exact | Call Exact | Argument F1 | Tool F1 | Checkpoint |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | llama8b_adap_tokmem_eoc_logit_bias_trial1_seed42 | 800 | 0.8500 | 0.4163 | 0.8415 | 0.9855 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama8b_adap_tokmem_eoc_logit_bias_trial1_seed42/round_2_tools_51_100.pt` |
| 2 | llama8b_adap_tokmem_eoc_logit_bias_trial2_seed42 | 800 | 0.8500 | 0.4163 | 0.8415 | 0.9855 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama8b_adap_tokmem_eoc_logit_bias_trial2_seed42/round_2_tools_51_100.pt` |
| 3 | llama8b_adap_tokmem_eoc_logit_bias_trial3_seed42 | 800 | 0.8500 | 0.4163 | 0.8415 | 0.9855 | `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/runs/llama8b_adap_tokmem_eoc_logit_bias_trial3_seed42/round_2_tools_51_100.pt` |
