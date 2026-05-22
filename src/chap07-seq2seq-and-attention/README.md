# 序列逆置（Seq2Seq）工程化补充说明

本目录示例是用 Seq2Seq 模型做字符串逆置任务（例如 `ABCDE -> EDCBA`）。

本次在不改变核心教学逻辑的前提下，对 `sequence_reversal-exercise.py` 做了小幅工程化增强：

1. 增加环境变量参数，避免反复改源码：
   - `SEQREV_SEED`
   - `SEQREV_TRAIN_STEPS`
   - `SEQREV_TRAIN_BATCH`
   - `SEQREV_SEQ_LEN`
   - `SEQREV_LOG_INTERVAL`
   - `SEQREV_TEST_BATCH`
   - `SEQREV_TEST_LEN`
   - `SEQREV_REPORT_OUT`
   - `SEQREV_SHOW_SAMPLES`
2. 固定随机种子（`random`/`numpy`/`tensorflow`）以保证可复现。
3. 增加评估与结果导出，自动生成 `sequence_reversal_report.json`。

## 运行示例（PowerShell）

```powershell
$env:SEQREV_SEED=42
$env:SEQREV_TRAIN_STEPS=200
$env:SEQREV_TRAIN_BATCH=32
$env:SEQREV_SEQ_LEN=20
$env:SEQREV_LOG_INTERVAL=50
$env:SEQREV_TEST_BATCH=32
$env:SEQREV_TEST_LEN=10
$env:SEQREV_REPORT_OUT="sequence_reversal_report.json"
$env:SEQREV_SHOW_SAMPLES=10
python .\sequence_reversal-exercise.py
```

## 输出结果

脚本运行后会输出训练日志、逆序准确率，并生成报告文件（默认）：

- `src/chap07-seq2seq-and-attention/sequence_reversal_report.json`

