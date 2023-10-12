# Low Latency

## Orca

其想做的就是给一堆请求, 这些请求是一个个prompt, 可能有长有短, 而且请求进入的时间也不同, 如何构造一个低延迟服务. 提出的贡献有2个, Iteration-level scheduling和selective batching.

### Iteration-Level Scheduling

把请求放在一个pool里, 做计算时由sheduler挑选一组batch的请求, 然后将其输入模型计算下一个token, 然后这样反复操作. 根据论文说法, 以往是一组batch算完为止, 这样的弊端是响应速度取决于整个batch内生成序列最长的那个, 会造成较大的延迟.

具体而言, 这里涉及到sheduler, 其满足FCFS (first-come-first-served) . 这个可以参考system的FIFO, 但这里先服务不代表其能先出来, 只是说其递归时会按照come的顺序进行.

### Selective Batching

这个部分是因为我们batch时其实很复杂, 有的请求还没被生成过token, 它们需要经过初始化过程; 经过生成的又有长有短, 这些复杂情况显然会对batch造成困难. 因为infer阶段, 我们倾向于使用KV cache来做下一个token的infer. 存在KV cache的, 又不等长的token没法打包成一个batch. 这里的做法是, 有一个kv cache池, 初始化后其kv会放在里面. batch时可以针对任意的都做batch, 但如果已经初始化了, 只取其最新的1个token, 如果没有初始化, 就取所有的token, 然后输入给LLM. 其在linear时和batch是无关的, 因此将上述token串起来通过线性层, 而后再分开, 分别给atten计算, 计算时如果初始化了则要从kv cache里取值以加快运算过程, 如果没有初始化则全算一遍. 算完以后更新kv cache, 如果end了要删去这个序列的cache, 否则要把新的kv加进去.

### 其他

还涉及到诸如pipeline并行之类的操作, 顺便pipeline并行对mini batch有所要求.
