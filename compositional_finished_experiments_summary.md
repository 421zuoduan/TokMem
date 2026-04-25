# Compositional 已完成实验汇总

生成时间：2026-04-26 01:20 CST

## 数据来源与判定

- `results/compositional/all_methods/`：当前主结果目录，也是 `scripts/compositional/run_paper_compositional_suite.sh --suite-name all_methods` 的最新运行目录。`task_status.json` 记录 255 个任务，其中 225 个 `success`，30 个 `pending`；4calls 为 165/165 success，10calls 为 60/90 success。
- 当前 suite 仍在运行。进程观察到 3 个 10calls 训练任务正在跑：`llama3b_tokmem_10calls_trial4_seed42`、`llama3b_tokmem_eoc_logit_bias_10calls_trial4_seed42`、`llama3b_tokmem_eoc_replace_head_10calls_trial4_seed42`。这些任务在 `task_status.json` 中仍计入 `pending`。
- `results/compositional/paper_compositional_mainline_bs/`：上一轮 mainline batch-size suite。`task_status.json` 记录 135 个任务，其中 130 个 `success`，5 个 `failed`。
- `compositional/runs/bs_probe_*`：当前只有 `run_config.json` 和 `evaluation.log` 启动记录，作为 batch-size probe 启动记录看待；该目录下没有 `SUCCESS`、`evaluation_results.json`、checkpoint 或最终指标。

指标取自每个成功 run 的 `evaluation_results.json` 最后一轮结果。表里的 `Train Params` 对应最后一个训练 round 的 active trainable parameter count；对 adaptation 方法，这是 tools 51-100 round 里 LoRA 冻结后的可训练参数量；ICL/RAG 为 0。`Tool Acc` 对应工具选择准确率，`Tool EM` 对应工具集合完全匹配，`Exact Acc` 对应完整调用 exact match，`Tool F1` 对应工具选择 F1，4calls/mainline 表里的 `Arg F1` 对应 `avg_f1_score`，10calls 快照沿用原列名且对应 `arguments_accuracy`，`Parse Err` 对应解析错误率。

## 当前 all_methods：4calls 完整组

这些组均已完成 5/5 trials。范围是 tools 51-100 / 4 calls。

| Model | Method | Done | Train Params | Tool Acc | Tool EM | Exact Acc | Tool F1 | Arg F1 | Parse Err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama1b | icl | 5/5 | 0 | 0.950 | 0.002 | 0.000 | 0.014 | 0.008 | 0.436 |
| llama1b | rag | 5/5 | 0 | 0.953 | 0.014 | 0.014 | 0.090 | 0.055 | 0.270 |
| llama1b | adap_tokmem | 5/5 | 204,800 | 0.999 | 0.919 | 0.632 | 0.979 | 0.831 | 0.220 |
| llama1b | adap_tokmem_eoc | 5/5 | 206,848 | 0.999 | 0.919 | 0.628 | 0.977 | 0.831 | 0.663 |
| llama1b | adap_tokmem_eoc_logit_bias | 5/5 | 411,748 | 0.999 | 0.929 | 0.646 | 0.989 | 0.843 | 0.034 |
| llama1b | adap_tokmem_eoc_replace_head | 5/5 | 411,748 | 0.999 | 0.944 | 0.658 | 0.988 | 0.850 | 0.032 |
| llama1b | lora | 5/5 | 851,968 | 0.981 | 0.569 | 0.544 | 0.809 | 0.780 | 0.000 |
| llama1b | tokmem | 5/5 | 102,400 | 0.982 | 0.558 | 0.397 | 0.812 | 0.662 | 0.863 |
| llama1b | tokmem_eoc | 5/5 | 104,448 | 0.986 | 0.606 | 0.443 | 0.846 | 0.707 | 1.508 |
| llama1b | tokmem_eoc_logit_bias | 5/5 | 206,898 | 0.991 | 0.705 | 0.469 | 0.899 | 0.730 | 0.473 |
| llama1b | tokmem_eoc_replace_head | 5/5 | 206,898 | 0.991 | 0.722 | 0.487 | 0.899 | 0.739 | 0.700 |
| llama3b | icl | 5/5 | 0 | 0.974 | 0.226 | 0.228 | 0.602 | 0.489 | 0.018 |
| llama3b | rag | 5/5 | 0 | 0.985 | 0.486 | 0.254 | 0.787 | 0.537 | 0.014 |
| llama3b | adap_tokmem | 5/5 | 307,200 | 1.000 | 0.958 | 0.696 | 0.994 | 0.870 | 0.018 |
| llama3b | adap_tokmem_eoc | 5/5 | 310,272 | 0.999 | 0.942 | 0.676 | 0.985 | 0.861 | 0.014 |
| llama3b | adap_tokmem_eoc_logit_bias | 5/5 | 617,572 | 1.000 | 0.958 | 0.722 | 0.992 | 0.880 | 0.008 |
| llama3b | adap_tokmem_eoc_replace_head | 5/5 | 617,572 | 1.000 | 0.958 | 0.724 | 0.990 | 0.880 | 0.006 |
| llama3b | lora | 5/5 | 2,293,760 | 0.998 | 0.926 | 0.698 | 0.977 | 0.871 | 0.000 |
| llama3b | tokmem | 5/5 | 153,600 | 0.979 | 0.474 | 0.366 | 0.785 | 0.645 | 13.084 |
| llama3b | tokmem_eoc | 5/5 | 156,672 | 0.982 | 0.506 | 0.388 | 0.817 | 0.696 | 2.016 |
| llama3b | tokmem_eoc_logit_bias | 5/5 | 310,322 | 0.981 | 0.482 | 0.326 | 0.809 | 0.614 | 29.982 |
| llama3b | tokmem_eoc_replace_head | 5/5 | 310,322 | 0.981 | 0.472 | 0.328 | 0.803 | 0.610 | 27.982 |
| llama8b | icl | 5/5 | 0 | 0.976 | 0.266 | 0.268 | 0.603 | 0.523 | 0.138 |
| llama8b | rag | 5/5 | 0 | 0.986 | 0.544 | 0.308 | 0.784 | 0.601 | 0.080 |
| llama8b | adap_tokmem | 5/5 | 409,600 | 0.999 | 0.940 | 0.670 | 0.984 | 0.855 | 0.018 |
| llama8b | adap_tokmem_eoc | 5/5 | 413,696 | 0.999 | 0.949 | 0.696 | 0.987 | 0.871 | 0.010 |
| llama8b | adap_tokmem_eoc_logit_bias | 5/5 | 823,396 | 0.999 | 0.936 | 0.694 | 0.983 | 0.866 | 0.032 |
| llama8b | adap_tokmem_eoc_replace_head | 5/5 | 823,396 | 0.999 | 0.928 | 0.688 | 0.982 | 0.863 | 0.030 |
| llama8b | lora | 5/5 | 3,407,872 | 0.998 | 0.930 | 0.736 | 0.977 | 0.883 | 0.000 |
| llama8b | tokmem | 5/5 | 204,800 | 0.984 | 0.570 | 0.446 | 0.841 | 0.712 | 0.564 |
| llama8b | tokmem_eoc | 5/5 | 208,896 | 0.986 | 0.602 | 0.478 | 0.858 | 0.738 | 0.158 |
| llama8b | tokmem_eoc_logit_bias | 5/5 | 413,746 | 0.987 | 0.618 | 0.506 | 0.873 | 0.733 | 0.330 |
| llama8b | tokmem_eoc_replace_head | 5/5 | 413,746 | 0.987 | 0.610 | 0.502 | 0.875 | 0.733 | 0.316 |

## 当前 all_methods：4calls 按实际 call 数分桶 F1

这些数值从每个 4calls 成功 run 的 `evaluation.log` 抽取。`F1` 对应 function call 序列 F1；`Tool F1` 对应工具选择 F1。表中数值是同一 model/method 的 5 个 trial 均值。

| Model | Method | Trials | 2-call Tool F1 | 3-call Tool F1 | 4-call Tool F1 | Tool F1 Avg | 2-call F1 | 3-call F1 | 4-call F1 | F1 Avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama1b | icl | 5/5 | 0.020 | 0.009 | 0.000 | 0.010 | 0.012 | 0.004 | 0.000 | 0.005 |
| llama1b | rag | 5/5 | 0.155 | 0.036 | 0.000 | 0.064 | 0.097 | 0.020 | 0.000 | 0.039 |
| llama1b | lora | 5/5 | 0.822 | 0.803 | 0.747 | 0.791 | 0.793 | 0.780 | 0.650 | 0.741 |
| llama1b | tokmem | 5/5 | 0.820 | 0.809 | 0.760 | 0.796 | 0.676 | 0.660 | 0.557 | 0.631 |
| llama1b | tokmem_eoc | 5/5 | 0.844 | 0.857 | 0.760 | 0.820 | 0.707 | 0.720 | 0.570 | 0.666 |
| llama1b | tokmem_eoc_logit_bias | 5/5 | 0.900 | 0.904 | 0.842 | 0.882 | 0.736 | 0.735 | 0.634 | 0.702 |
| llama1b | tokmem_eoc_replace_head | 5/5 | 0.898 | 0.907 | 0.822 | 0.876 | 0.744 | 0.746 | 0.623 | 0.705 |
| llama1b | adap_tokmem | 5/5 | 0.979 | 0.983 | 0.949 | 0.970 | 0.833 | 0.836 | 0.754 | 0.808 |
| llama1b | adap_tokmem_eoc | 5/5 | 0.976 | 0.982 | 0.954 | 0.971 | 0.826 | 0.843 | 0.755 | 0.808 |
| llama1b | adap_tokmem_eoc_logit_bias | 5/5 | 0.990 | 0.989 | 0.972 | 0.984 | 0.845 | 0.851 | 0.762 | 0.819 |
| llama1b | adap_tokmem_eoc_replace_head | 5/5 | 0.991 | 0.987 | 0.977 | 0.985 | 0.861 | 0.847 | 0.777 | 0.828 |
| llama3b | icl | 5/5 | 0.595 | 0.612 | 0.569 | 0.592 | 0.487 | 0.494 | 0.452 | 0.478 |
| llama3b | rag | 5/5 | 0.802 | 0.770 | 0.815 | 0.796 | 0.552 | 0.528 | 0.473 | 0.518 |
| llama3b | lora | 5/5 | 0.973 | 0.987 | 0.924 | 0.961 | 0.879 | 0.873 | 0.778 | 0.843 |
| llama3b | tokmem | 5/5 | 0.791 | 0.782 | 0.768 | 0.780 | 0.656 | 0.634 | 0.650 | 0.647 |
| llama3b | tokmem_eoc | 5/5 | 0.817 | 0.816 | 0.837 | 0.823 | 0.686 | 0.703 | 0.724 | 0.704 |
| llama3b | tokmem_eoc_logit_bias | 5/5 | 0.803 | 0.816 | 0.789 | 0.803 | 0.619 | 0.618 | 0.526 | 0.588 |
| llama3b | tokmem_eoc_replace_head | 5/5 | 0.805 | 0.807 | 0.746 | 0.786 | 0.619 | 0.612 | 0.501 | 0.577 |
| llama3b | adap_tokmem | 5/5 | 0.996 | 0.994 | 0.972 | 0.987 | 0.882 | 0.866 | 0.800 | 0.849 |
| llama3b | adap_tokmem_eoc | 5/5 | 0.982 | 0.989 | 0.979 | 0.983 | 0.874 | 0.862 | 0.739 | 0.825 |
| llama3b | adap_tokmem_eoc_logit_bias | 5/5 | 0.989 | 0.994 | 0.992 | 0.992 | 0.882 | 0.885 | 0.813 | 0.860 |
| llama3b | adap_tokmem_eoc_replace_head | 5/5 | 0.989 | 0.994 | 0.971 | 0.985 | 0.882 | 0.885 | 0.813 | 0.860 |
| llama8b | icl | 5/5 | 0.577 | 0.636 | 0.523 | 0.579 | 0.496 | 0.552 | 0.497 | 0.515 |
| llama8b | rag | 5/5 | 0.790 | 0.782 | 0.741 | 0.771 | 0.601 | 0.607 | 0.551 | 0.586 |
| llama8b | lora | 5/5 | 0.977 | 0.981 | 0.939 | 0.966 | 0.892 | 0.877 | 0.857 | 0.875 |
| llama8b | tokmem | 5/5 | 0.843 | 0.851 | 0.727 | 0.807 | 0.727 | 0.707 | 0.625 | 0.686 |
| llama8b | tokmem_eoc | 5/5 | 0.869 | 0.856 | 0.777 | 0.834 | 0.746 | 0.735 | 0.682 | 0.721 |
| llama8b | tokmem_eoc_logit_bias | 5/5 | 0.881 | 0.875 | 0.775 | 0.844 | 0.743 | 0.733 | 0.643 | 0.706 |
| llama8b | tokmem_eoc_replace_head | 5/5 | 0.888 | 0.873 | 0.779 | 0.847 | 0.749 | 0.727 | 0.643 | 0.706 |
| llama8b | adap_tokmem | 5/5 | 0.981 | 0.991 | 0.949 | 0.974 | 0.853 | 0.866 | 0.766 | 0.828 |
| llama8b | adap_tokmem_eoc | 5/5 | 0.986 | 0.990 | 0.970 | 0.982 | 0.876 | 0.870 | 0.821 | 0.856 |
| llama8b | adap_tokmem_eoc_logit_bias | 5/5 | 0.982 | 0.985 | 0.965 | 0.977 | 0.872 | 0.868 | 0.780 | 0.840 |
| llama8b | adap_tokmem_eoc_replace_head | 5/5 | 0.982 | 0.983 | 0.965 | 0.977 | 0.872 | 0.866 | 0.756 | 0.831 |

## 当前 all_methods：10calls 已完成 trial 快照

这些组仍在 suite 计划内推进；表中均值只基于当前已成功 trials。范围是 tools 1-50 / 4 calls 后接 tools 51-100 / 10 calls，表中指标取最后一轮 tools 51-100 / 10 calls。当前 10calls 已完成 60/90 trials。

| Model | Method | Done | Tool Acc | Tool EM | Exact Acc | Tool F1 | Arg F1 | Parse Err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama1b | tokmem | 4/5 | 0.977 | 0.473 | 0.245 | 0.885 | 0.741 | 0.794 |
| llama1b | tokmem_eoc_logit_bias | 4/5 | 0.985 | 0.572 | 0.296 | 0.926 | 0.792 | 0.699 |
| llama1b | tokmem_eoc_replace_head | 4/5 | 0.984 | 0.573 | 0.290 | 0.918 | 0.782 | 3.587 |
| llama1b | adap_tokmem | 4/5 | 0.997 | 0.772 | 0.375 | 0.972 | 0.842 | 0.521 |
| llama1b | adap_tokmem_eoc_logit_bias | 4/5 | 0.998 | 0.840 | 0.412 | 0.984 | 0.870 | 0.393 |
| llama1b | adap_tokmem_eoc_replace_head | 4/5 | 0.998 | 0.849 | 0.418 | 0.986 | 0.869 | 0.105 |
| llama3b | tokmem | 3/5 | 0.966 | 0.347 | 0.188 | 0.821 | 0.684 | 3.431 |
| llama3b | tokmem_eoc_logit_bias | 3/5 | 0.978 | 0.448 | 0.245 | 0.887 | 0.778 | 4.802 |
| llama3b | tokmem_eoc_replace_head | 3/5 | 0.979 | 0.465 | 0.238 | 0.890 | 0.778 | 3.891 |
| llama3b | adap_tokmem | 3/5 | 0.999 | 0.887 | 0.467 | 0.988 | 0.867 | 0.057 |
| llama3b | adap_tokmem_eoc_logit_bias | 3/5 | 0.999 | 0.909 | 0.463 | 0.993 | 0.881 | 0.027 |
| llama3b | adap_tokmem_eoc_replace_head | 3/5 | 0.999 | 0.905 | 0.472 | 0.993 | 0.889 | 0.133 |
| llama8b | tokmem | 3/5 | 0.969 | 0.313 | 0.172 | 0.845 | 0.648 | 1.226 |
| llama8b | tokmem_eoc_logit_bias | 3/5 | 0.976 | 0.403 | 0.206 | 0.883 | 0.702 | 0.618 |
| llama8b | tokmem_eoc_replace_head | 3/5 | 0.978 | 0.408 | 0.219 | 0.891 | 0.733 | 0.486 |
| llama8b | adap_tokmem | 3/5 | 0.998 | 0.874 | 0.400 | 0.987 | 0.832 | 0.066 |
| llama8b | adap_tokmem_eoc_logit_bias | 3/5 | 0.998 | 0.869 | 0.395 | 0.985 | 0.832 | 0.040 |
| llama8b | adap_tokmem_eoc_replace_head | 3/5 | 0.998 | 0.872 | 0.402 | 0.985 | 0.831 | 0.030 |

## 上一轮 mainline_bs：完整结果索引

`results/compositional/paper_compositional_mainline_bs/summary.md` 已包含 `icl`、`rag`、`lora`、`tokmem`、`tokmem_eoc`、`tokmem_eoc_logit_bias` 在 llama1b/3b/8b 上的 5-trial 均值。该 suite 的基础组全部成功。

维护路径里，`tokmem_eoc_logit_bias` 的 mean exact accuracy 为：

| Model | Method | Done | Tool Acc | Tool EM | Exact Acc | Tool F1 | Arg F1 | Parse Err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama1b | tokmem_eoc_logit_bias | 5/5 | 0.991 | 0.706 | 0.478 | 0.899 | 0.738 | 0.650 |
| llama3b | tokmem_eoc_logit_bias | 5/5 | 0.984 | 0.510 | 0.362 | 0.836 | 0.650 | 9.436 |
| llama8b | tokmem_eoc_logit_bias | 5/5 | 0.988 | 0.644 | 0.515 | 0.879 | 0.762 | 0.361 |

该 suite 里额外 adaptation 组的成功均值如下：

| Model | Method | Done | Tool Acc | Tool EM | Exact Acc | Tool F1 | Arg F1 | Parse Err |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| llama1b | adap_tokmem | 5/5 | 0.999 | 0.919 | 0.634 | 0.979 | 0.834 | 0.147 |
| llama1b | adap_tokmem_eoc | 5/5 | 0.999 | 0.914 | 0.627 | 0.977 | 0.831 | 0.055 |
| llama3b | adap_tokmem | 5/5 | 1.000 | 0.957 | 0.699 | 0.993 | 0.865 | 0.052 |
| llama3b | adap_tokmem_eoc | 5/5 | 0.999 | 0.942 | 0.712 | 0.985 | 0.874 | 0.013 |
| llama3b | adap_tokmem_eoc_logit_bias | 5/5 | 0.999 | 0.950 | 0.701 | 0.988 | 0.870 | 0.013 |
| llama8b | adap_tokmem | 5/5 | 1.000 | 0.957 | 0.664 | 0.990 | 0.855 | 0.018 |
| llama8b | adap_tokmem_eoc | 5/5 | 0.999 | 0.947 | 0.674 | 0.986 | 0.858 | 0.012 |
| llama8b | adap_tokmem_eoc_logit_bias | 5/5 | 1.000 | 0.960 | 0.748 | 0.991 | 0.897 | 0.006 |

`paper_compositional_mainline_bs` 里 `llama1b_adap_tokmem_eoc_logit_bias_trial1-5_seed42` 的 5 个 trial 均以 exit code 1 结束。

## 直接观察

- 当前 `all_methods` 里，4calls 的完整组已经覆盖 llama1b/3b/8b 的 `lora`、TokMem、EOC、EOC+logit-bias、replace-head、adaptation 变体。
- 在 4calls 完整组里，`adap_tokmem_eoc_logit_bias` 的 Exact Acc 分别是 llama1b 0.646、llama3b 0.722、llama8b 0.694；同组 `tokmem_eoc_logit_bias` 分别是 0.469、0.326、0.506。
- 10calls 当前完成 60/90 trials。llama1b 各组已经到 4/5，llama3b 和 llama8b 各组大多到 3/5；当前仍在运行的 3 个训练任务推进 llama3b TokMem trial4。
- 10calls 已完成 trial 中 adaptation 系列的 Exact Acc 高于同模型的 TokMem 系列；当前最高的 10calls Exact Acc 是 `llama3b_adap_tokmem_eoc_replace_head` 0.472，其次是 `llama3b_adap_tokmem` 0.467 和 `llama3b_adap_tokmem_eoc_logit_bias` 0.463。
- `replace_head` 出现在当前结果中，属于额外对照组；维护路径解读仍应优先看 `eoc` 与 `logit_bias` 系列。
