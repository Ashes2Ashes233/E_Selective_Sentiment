# E题三模态选择性情感预测方法与服务器迁移说明

完整可编辑版本见同目录DOCX。模型使用附件2对齐版，训练/验证/测试为3395/728/727。输入为text_bert(3×50)、audio(50×74)、vision(50×35)，BERT输出50×768；三路适配到96维后动态门控，循环融合和有效位置聚合，输出三分类概率与[-3,3]情感强度。

## 当前实测

测试集 Accuracy=0.6768，Macro-F1=0.6209，MAE=0.6478，Pearson=0.6637。验证集 Accuracy=0.6250，Macro-F1=0.5918，MAE=0.5963，Pearson=0.6543。

## 服务器迁移

可以迁移。必须带selective_sentiment/*.py、requirements-runtime.txt、附件2 aligned_50.pkl、pretrained/bert/和可选的outputs/best.pt。若还要跑问题1和问题3视频定位，再带附件1视频、label-100.xlsx、附件3/4、pretrained/aligner/。Linux上先按服务器CUDA驱动安装兼容PyTorch，再安装其余依赖；设置HF_HUB_OFFLINE=1时需提前复制两套预训练权重。训练示例：python train.py --data /workspace/E题数据/E题数据 --output outputs_long --epochs 50 --batch-size 64 --gate soft。

完整架构、维度、指标公式、目录和论文边界见DOCX。
