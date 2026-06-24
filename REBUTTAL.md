# Rebuttal 大纲

## 实验

方法本身:

1. 对比实验: 不用额外的 adapter, memory token 解码时直接约束在 memory bank 里, 对比有 adapter 的效果
2. 对比实验: adapter logits 和 lm head logits 直接相加的实验, 不对 adapter logits 做额外操作
3. 对比实验: AR loss 不影响 adapter, adapter 只受到 routing loss 更新

实验设置

1. 使用合成数据集: 尝试在 TaskBench, APIGen, $\tau$-bench, StableToolBench, Berkeley Function Calling Leaderboard 上进行实验
2. 泛化性实验: 在训练上只跑过 4call 的模型在 10call 上测试
3. EOC 边界预测准确率实验: 补一个 \<EOC\> token 的 F1 分数, 证明边界预测准确
4. 效果提升来源实验: 无 adaptation 效果提升主要原因是 transition aware, 而非参数量增加
    1. 汇报下 procedure 格式错误的比例, 证明去掉格式错误的 procedure, 我们的方法依然减少了 transition error; 
    2. 考虑到现有统计方法在格式错误时会判定参数生成错误, 实际存在格式错误但参数生成正确的情况, 如果上面的结果无法说明, 拿大模型逐个 procedure 分析再汇报一个结果
5. TokMem 的实验里, procedure 顺序错了但 set 一样仍会算是对的, 统计下这类错误的比例
6. 消融实验: adapter 使用 linear 和 mlp 两种结构的效果对比
7. 修正效果对比: adapter 修正 memory token logits 前后词表分布的熵对比和修正准确率对比
8. memory 更新: 在已经训练好的 memory bank 基础上, 加入新的 memory token, 看原有 memory token 路由准确率的变化

额外分析:

1. 错误类型分析: 对比 TapMem 和 TokMem 在不同错误类型上的降低比例, 证明 TapMem 主要减少的是 transition-related errors, 比如 wrong eoc, extra eoc, missing procedure, extra procedure, correct tool but wrong arguments, 
2. 


无法回应的质疑:

1. adapter 输出空间是 memory token + eoc token 的词表空间, 效果会不会更好
