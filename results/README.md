# 实验结果

这里只存经核验、无隐私的小型汇总结果。不把合成测试通过当作实验结果，
开发集结果与正式共同集合 test 结果必须分开标识。

每份结果须注明 task、protocol/schema、split/cohort、特征与代码版本、种子、评价单位和真实运行范围。完整预测与 checkpoint 留在受控运行包。

当前结果：

- `baselines/fatigue_can/gru_v1_train_val_20260908.md`：UL-DD CAN-only
  complete-8 train/val 开发实验；后续审计已确认与视频+CAN共同 train/val
  集合完全一致，可用作该集合上的单模态对照；未访问 test。
