# GPU 大显存进程监测

`tmp/monitor_gpu_memory_processes.py` 持续查询 NVIDIA compute
process，并把跨 GPU 汇总显存严格大于阈值的进程写入一个 CSV。默认阈值为
5 GiB，默认每 10 秒查询一次。脚本、CSV 和后台运行日志都放在项目根目录的
`tmp/` 中，该目录不会被 Git 跟踪。

```bash
nohup python tmp/monitor_gpu_memory_processes.py \
    --threshold-gib 5 \
    --poll-seconds 10 \
    --output tmp/gpu_memory_processes.csv \
    >tmp/gpu_process_monitor.log 2>&1 &
```

CSV 中的 `active` 行代表进程仍在执行，因此结束时间和持续时间暂时为空。进程
结束后，同一行会更新成 `finished` 并补全 `end_time`、`duration_seconds` 和
`duration`。进程开始时间取自 Linux `/proc`，结束时间是监测器首次发现进程
退出的时间，误差不超过正常轮询间隔；如果监测器中途停止，误差会包含停机
时间。进程暂时降到 5 GiB 以下不会提前结束记录，监测器会一直跟踪到该进程
真正退出。

同一个输出文件也用于恢复仍在执行的记录，因此重启时应继续传入原来的
`--output`。脚本会为输出文件创建一个隐藏锁文件，防止两个监测器同时写入。
