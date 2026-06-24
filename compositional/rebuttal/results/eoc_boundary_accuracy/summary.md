# Generated EOC Boundary Accuracy

- all_methods status: `/data/shilong/tokmem/results/compositional/all_methods/task_status.json`
- paper head status: `/data/shilong/tokmem/results/compositional/paper_compositional_head_8gpu/task_status.json`
- llama3b final checkpoint suite: `/data/shilong/tokmem/results/compositional/llama3b_tokmem_family_4calls_seed42_3x_20260425_214658`
- max new tokens: `512`
- eval batch size: `8`

## Aggregate

| Model | Method | Trials | Samples | EOC Count Exact | EOC Precision | EOC Recall | EOC F1 | Expected EOC | Predicted EOC | Missing EOC | Extra EOC | Malformed Boundary Rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama1b | eoc | 5 | 2500 | 0.8804 | 0.9892 | 0.9503 | 0.9694 | 2.5840 | 2.4824 | 0.1280 | 0.0264 | 0.0348 |
| llama1b | eoc_logit_bias | 3 | 1500 | 0.9040 | 0.9801 | 0.9636 | 0.9718 | 2.5840 | 2.5407 | 0.0940 | 0.0507 | 0.0207 |
| llama8b | eoc | 5 | 2500 | 0.9260 | 0.9890 | 0.9714 | 0.9801 | 2.5840 | 2.5380 | 0.0720 | 0.0260 | 0.0660 |
| llama8b | eoc_logit_bias | 3 | 1500 | 0.9420 | 0.9838 | 0.9868 | 0.9853 | 2.5840 | 2.5920 | 0.0340 | 0.0420 | 0.0540 |
| llama3b | eoc | 3 | 1500 | 0.7893 | 0.6748 | 0.9231 | 0.7797 | 2.5840 | 3.5347 | 0.1973 | 1.1480 | 0.1733 |
| llama3b | eoc_logit_bias | 3 | 1500 | 0.8147 | 0.6024 | 0.9303 | 0.7313 | 2.5840 | 3.9907 | 0.1800 | 1.5867 | 0.1847 |

## Per Trial

### llama1b / eoc

| Trial | Samples | EOC Count Exact | EOC Precision | EOC Recall | EOC F1 | Predicted EOC | Missing EOC | Extra EOC | Run |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 500 | 0.8840 | 0.9944 | 0.9567 | 0.9751 | 2.4860 | 0.1120 | 0.0140 | `llama1b_tokmem_eoc_trial1_seed42` |
| 2 | 500 | 0.8840 | 0.9927 | 0.9481 | 0.9699 | 2.4680 | 0.1340 | 0.0180 | `llama1b_tokmem_eoc_trial2_seed42` |
| 3 | 500 | 0.8920 | 0.9873 | 0.9659 | 0.9765 | 2.5280 | 0.0880 | 0.0320 | `llama1b_tokmem_eoc_trial3_seed42` |
| 4 | 500 | 0.8660 | 0.9789 | 0.9319 | 0.9548 | 2.4600 | 0.1740 | 0.0500 | `llama1b_tokmem_eoc_trial4_seed42` |
| 5 | 500 | 0.8760 | 0.9927 | 0.9489 | 0.9703 | 2.4700 | 0.1320 | 0.0180 | `llama1b_tokmem_eoc_trial5_seed42` |

### llama1b / eoc_logit_bias

| Trial | Samples | EOC Count Exact | EOC Precision | EOC Recall | EOC F1 | Predicted EOC | Missing EOC | Extra EOC | Run |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 500 | 0.8900 | 0.9778 | 0.9567 | 0.9671 | 2.5280 | 0.1120 | 0.0560 | `llama1b_tokmem_eoc_logit_bias_trial1_seed42` |
| 2 | 500 | 0.9000 | 0.9734 | 0.9636 | 0.9685 | 2.5580 | 0.0940 | 0.0680 | `llama1b_tokmem_eoc_logit_bias_trial2_seed42` |
| 3 | 500 | 0.9220 | 0.9890 | 0.9706 | 0.9797 | 2.5360 | 0.0760 | 0.0280 | `llama1b_tokmem_eoc_logit_bias_trial3_seed42` |

### llama8b / eoc

| Trial | Samples | EOC Count Exact | EOC Precision | EOC Recall | EOC F1 | Predicted EOC | Missing EOC | Extra EOC | Run |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 500 | 0.9260 | 0.9890 | 0.9714 | 0.9801 | 2.5380 | 0.0720 | 0.0260 | `llama8b_tokmem_eoc_trial1_seed42` |
| 2 | 500 | 0.9260 | 0.9890 | 0.9714 | 0.9801 | 2.5380 | 0.0720 | 0.0260 | `llama8b_tokmem_eoc_trial2_seed42` |
| 3 | 500 | 0.9260 | 0.9890 | 0.9714 | 0.9801 | 2.5380 | 0.0720 | 0.0260 | `llama8b_tokmem_eoc_trial3_seed42` |
| 4 | 500 | 0.9260 | 0.9890 | 0.9714 | 0.9801 | 2.5380 | 0.0720 | 0.0260 | `llama8b_tokmem_eoc_trial4_seed42` |
| 5 | 500 | 0.9260 | 0.9890 | 0.9714 | 0.9801 | 2.5380 | 0.0720 | 0.0260 | `llama8b_tokmem_eoc_trial5_seed42` |

### llama8b / eoc_logit_bias

| Trial | Samples | EOC Count Exact | EOC Precision | EOC Recall | EOC F1 | Predicted EOC | Missing EOC | Extra EOC | Run |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 500 | 0.9420 | 0.9838 | 0.9868 | 0.9853 | 2.5920 | 0.0340 | 0.0420 | `llama8b_tokmem_eoc_logit_bias_trial1_seed42` |
| 2 | 500 | 0.9420 | 0.9838 | 0.9868 | 0.9853 | 2.5920 | 0.0340 | 0.0420 | `llama8b_tokmem_eoc_logit_bias_trial2_seed42` |
| 3 | 500 | 0.9420 | 0.9838 | 0.9868 | 0.9853 | 2.5920 | 0.0340 | 0.0420 | `llama8b_tokmem_eoc_logit_bias_trial3_seed42` |

### llama3b / eoc

| Trial | Samples | EOC Count Exact | EOC Precision | EOC Recall | EOC F1 | Predicted EOC | Missing EOC | Extra EOC | Run |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 500 | 0.7880 | 0.4387 | 0.9257 | 0.5953 | 5.4520 | 0.1900 | 3.0580 | `tokmem_eoc_llama_3b_4calls_seed42_3x_20260425_214658_trial1` |
| 2 | 500 | 0.8420 | 0.9728 | 0.9404 | 0.9563 | 2.4980 | 0.1520 | 0.0660 | `tokmem_eoc_llama_3b_4calls_seed42_3x_20260425_214658_trial2` |
| 3 | 500 | 0.7380 | 0.8794 | 0.9033 | 0.8912 | 2.6540 | 0.2500 | 0.3200 | `tokmem_eoc_llama_3b_4calls_seed42_3x_20260425_214658_trial3` |

### llama3b / eoc_logit_bias

| Trial | Samples | EOC Count Exact | EOC Precision | EOC Recall | EOC F1 | Predicted EOC | Missing EOC | Extra EOC | Run |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 500 | 0.7940 | 0.6070 | 0.9311 | 0.7349 | 3.9640 | 0.1780 | 1.5580 | `tokmem_eoc_logit_bias_llama_3b_4calls_seed42_3x_20260425_214658_trial1` |
| 2 | 500 | 0.7580 | 0.4248 | 0.8986 | 0.5769 | 5.4660 | 0.2620 | 3.1440 | `tokmem_eoc_logit_bias_llama_3b_4calls_seed42_3x_20260425_214658_trial2` |
| 3 | 500 | 0.8920 | 0.9772 | 0.9613 | 0.9692 | 2.5420 | 0.1000 | 0.0580 | `tokmem_eoc_logit_bias_llama_3b_4calls_seed42_3x_20260425_214658_trial3` |
