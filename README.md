# E题三模态选择性情感预测

面向比赛的独立改造版。继承 Visual-Selective-VIO 的策略网络、门控融合与循环状态思路；原始VIO代码保留不变。本目录完成问题1特征提取、问题2局部缺失预测、问题3解释与原视频证据定位。

## 快速运行

Python 3.12。默认数据目录为本项目同级的 `E题数据/E题数据`，使用对齐版。第一次准备：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_models.py bert aligner
.\.venv\Scripts\python.exe run_all.py --epochs 8
```

GPU建议安装与显卡兼容的CUDA版PyTorch；本次使用PyTorch 2.7.1+cu128。GPU不是必需，但冻结BERT和ASR在CPU上较慢。当前工作区的`.venv`已经配置；实际运行命令可直接使用该环境。

已有 `outputs/best.pt` 时，`run_all.py`跳过训练。`--retrain`明确要求重新训练。专项推理无需重新训练：

```powershell
.\.venv\Scripts\python.exe predict.py
.\.venv\Scripts\python.exe make_report.py
```

训练开关：`train.py --gate soft|hard|fixed --epochs 8 --batch-size 32 --seed 2026`。默认soft三模态门控，hard是独立二分类Gumbel训练、确定性推理；不默认执行多模型对照。若需修改数据目录，分别给 train.py、media.py、predict.py 指定 `--data`。

## 输出

- `outputs/q1/features/`：100条原视频的三模态特征，含有效掩码、词元编号及时间范围。
- `outputs/q1/sample_summary.csv`：全样本提取统计；失败样本保留且明确标记。
- `outputs/q1/alignment/`：CTC强制对齐及WordPiece映射日志。
- `outputs/q2_predictions.csv`：附件3全部30条对齐版样本预测。
- `outputs/q3_predictions_explanations.csv`：附件4全部20条预测、模态作用和关键证据。
- `outputs/cards/`、`outputs/keyframes/`：解释卡与原视频帧。
- `outputs/metrics.json`、`robustness.csv`、`training_history.csv`：实际训练、验证与缺失情景结果。
- `outputs/结果说明.md`、`结果总览.html`：可用于论文组织的结果说明和图表入口。
- `E题提交材料.zip`：程序、小型任务模型、自生成特征和结果，总量校验≤50MB。

## 方法与数据规则

附件2按3395/728/727既有划分。标签从连续值导出：负数Negative，零Neutral，正数Positive。分类输出和强度输出为两个头，因此个别样本可能存在符号与预测类别不一致；保留原始输出以便分析，不通过测试标签调整。

附件3对齐版没有`text`，因此三套附件统一使用`text_bert`经同一个冻结BERT编码。文本缺失在编码前实施，并在投影后再次施加可用性掩码；不从raw_text恢复遮挡。音视频标准化仅使用训练集有效观测。真实可用性根据掩码、特殊词元和全零连续行推断，并不声称能够完美区分真实静默、提取失败与人为缺失。

模型：三路96维投影，上一时刻192维GRU状态与当前特征共同输入门控，融合后形成新状态，注意力池化后输出三分类与[-3,3]强度。训练目标为Huber+0.7加权交叉熵，默认8轮，验证早停；不采用节算惩罚，不宣称实际减少BERT算力。问题2、3共用模型，问题3增加干预解释。

问题1文本768维、声学32维、视觉35维；这些自定义特征不冒充附件2原始提取器。声学使用MFCC及频谱/能量，视觉使用面部区域DCT/色彩/运动代理。模型不利用附件1增加情感监督训练样本。

解释使用完整预测与遮挡后的固定类别概率差、强度差。平均门控仅是选择行为。原始特征不附时间戳，问题3以强制对齐的词元时间映射对应位置，保留映射假设；不把50位置视为50个等时长区间。CTC失败时保留样本并明确标注近似回退。

## 预训练权重与复现体积

不在50MB提交包内打包通用BERT/ASR权重，运行前按download_models.py获取。下载来源和本次SHA256记录在pretrained各目录的download_manifest.json。默认下载器使用官方仓库main，因此远端未来发生变更时应核对原清单并获取匹配版本。冻结权重不参与情感训练；小型任务权重在outputs/best.pt。

## 来源

- Yang, Chen and Kim. Efficient Deep Visual and Inertial Odometry with Adaptive Visual Modality Selection. ECCV 2022. https://arxiv.org/abs/2205.06187
- 原作者代码：https://github.com/mingyuyng/Visual-Selective-VIO
- BERT：https://huggingface.co/google-bert/bert-base-uncased
- ASR强制对齐发射模型：https://huggingface.co/facebook/wav2vec2-base-960h

本实现为快速比赛版本，不将单次运行作为最优性、统计显著性或现实因果关系证明。报告中的所有性能数字从本地实际结果读取。
