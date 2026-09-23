"""Export figures, a factual results memo, and a <=50MB submission archive."""
import argparse
import html
import json
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT, save_json


def run(output):
    out = Path(output)
    figures = out / 'figures'; figures.mkdir(exist_ok=True)
    metrics = json.loads((out / 'metrics.json').read_text(encoding='utf-8'))
    history = pd.read_csv(out / 'training_history.csv')
    robustness = pd.read_csv(out / 'robustness.csv')
    q1 = pd.read_csv(out / 'q1' / 'sample_summary.csv')
    q2 = pd.read_csv(out / 'q2_predictions.csv')
    q3 = pd.read_csv(out / 'q3_predictions_explanations.csv')
    assert len(q1) == 100 and len(q2) == 30 and len(q3) == 20, 'Incomplete required output'
    for df in (q2, q3):
        assert df.sample_id.is_unique and df.predicted_score.between(-3, 3).all()
        probs = df[['prob_Negative', 'prob_Neutral', 'prob_Positive']].values
        assert np.isfinite(probs).all() and np.allclose(probs.sum(1), 1, atol=1e-5)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.epoch, history.validation_mae, 'o-', label='Complete validation')
    ax.plot(history.epoch, history.missing_mae, 'o-', label='Corrupted validation')
    ax.set(xlabel='Epoch', ylabel='MAE'); ax.legend(); fig.tight_layout()
    fig.savefig(figures / 'training.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for modality, group in robustness[robustness.position == 'middle'].groupby('modality'):
        group = group.sort_values('rate')
        for ax, metric in zip(axes, ('mae', 'macro_f1')):
            ax.plot([0] + group.rate.tolist(), [metrics['valid'][metric]] + group[metric].tolist(), 'o-', label=modality)
            ax.set(xlabel='Missing span fraction', ylabel=metric); ax.legend()
    fig.tight_layout(); fig.savefig(figures / 'robustness.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = np.asarray(metrics['test']['confusion_matrix'])
    image = ax.imshow(cm, cmap='Blues')
    for i in range(3):
        for j in range(3): ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    ax.set(xticks=range(3), yticks=range(3), xticklabels=['Neg', 'Neutral', 'Pos'],
           yticklabels=['Neg', 'Neutral', 'Pos'], xlabel='Predicted', ylabel='True')
    fig.colorbar(image, ax=ax); fig.tight_layout(); fig.savefig(figures / 'confusion.png', dpi=160); plt.close(fig)
    effect = pd.read_csv(out / 'q3_window_effects.csv')
    first_id = q3.sample_id.iloc[0]
    local = effect[(effect.sample_id == first_id) & (effect.kind == 'window')]
    fig, ax = plt.subplots(figsize=(8, 4))
    for m, g in local.groupby('modality'): ax.plot(g.start_index, g.delta_target_probability, 'o-', label=m)
    ax.axhline(0, color='gray', linewidth=.7)
    ax.set(xlabel='Window start position', ylabel='Target probability drop'); ax.legend()
    fig.tight_layout(); fig.savefig(figures / 'local_evidence.png', dpi=160); plt.close(fig)
    # One concrete Q1 correspondence table, with local feature vectors and video frame.
    first_ok = q1[q1.status == 'ok'].iloc[0]
    sid = first_ok.sample_id
    mapping = json.loads((out / 'q1' / 'alignment' / f'{sid}.json').read_text(encoding='utf-8'))
    values = np.load(out / 'q1' / 'features' / f'{sid}.npz')
    example = []
    for index, (left, right) in enumerate(mapping['token_spans']):
        if right <= left: continue
        a, b = mapping['token_offsets'][index]
        example.append({'position': index, 'text_piece': mapping['original_text'][a:b],
                        'start_seconds': left, 'end_seconds': right,
                        'audio_mfcc0': float(values['audio'][index, 0]),
                        'visual_dct0': float(values['vision'][index, 0]),
                        'text_feature_norm': float(np.linalg.norm(values['text'][index].astype(np.float32)))})
    pd.DataFrame(example).to_csv(out / 'q1' / 'alignment_example.csv', index=False, encoding='utf-8-sig')
    from common import DATA
    from predict import save_frame
    video_id, clip_id = sid.split('$_$')
    video = next(p for p in DATA.rglob(f'{clip_id}.mp4') if p.parent.name == video_id and '附件1' in str(p))
    save_frame(video, (example[0]['start_seconds'] + example[0]['end_seconds']) / 2, out / 'q1' / 'example_frame.jpg')
    fidelity = pd.read_csv(out / 'explanation_fidelity.csv')
    val, test = metrics['valid'], metrics['test']
    text_missing = robustness[(robustness.modality == 'text') & (robustness.rate == 0.7)].iloc[0]
    errors = pd.read_csv(out / 'valid_predictions.csv')
    errors['absolute_error'] = (errors.true_score - errors.predicted_score).abs()
    errors.sort_values('absolute_error', ascending=False).head(20).to_csv(
        out / 'validation_error_cases.csv', index=False, encoding='utf-8-sig')
    rows = '\n'.join(f'|{name}|{d["accuracy"]:.4f}|{d["macro_f1"]:.4f}|{d["mae"]:.4f}|{d["pearson"]:.4f}|'
                     for name, d in [('验证集', val), ('测试集', test)])
    failed = int((q1.status != 'ok').sum())
    fallback = int(q1.get('alignment', pd.Series(dtype=str)).fillna('').str.contains('fallback').sum())
    memo = f'''# E题快速实现结果说明

本实现继承 Visual-Selective-VIO 的策略选择与循环状态融合思想，改为文本、语音、视觉三路输入、缺失感知门控及片段级情感预测。以下数字来自本次实际运行；没有进行多种子或大规模调参。

## 问题1

全部100条样本均已保留，成功提取 {100-failed} 条，失败保留掩码 {failed} 条，粗略均匀对齐回退 {fallback} 条。文本采用冻结BERT最后层768维，声学32维，视觉35维。音频与视觉按CTC词级时间区间聚合，再映射至WordPiece位置。实际有效长度、时长、面部检测率与对齐状态见 q1/sample_summary.csv。

视觉特征是面部区域外观和运动的简单代理，不是经过情感监督训练的面部动作单元；未检测到人脸时采用整帧并记录检测率。这是快速方案的主要局限。问题1的自生成特征不与附件2的74/35维特征混用。

典型样本 {sid} 的逐位置对应见 q1/alignment_example.csv；原视频帧见 q1/example_frame.jpg。全部特征与完整时间映射分别在 q1/features 与 q1/alignment。

## 问题2

使用附件2既有3395/728/727训练、验证、测试划分。音视频标准化仅用训练集统计；文本统一通过冻结BERT重新编码。训练随机遮挡有效位置中的连续区间，文本在BERT编码前屏蔽，未使用raw_text恢复缺失。模型使用96维三路投影、192维GRU状态，策略读取当前可用特征、可用性与上一时刻隐状态。分类和回归损失联合训练，最佳轮数为 {metrics['best_epoch']}。

|划分|Accuracy|Macro F1|MAE|Pearson|
|---|---:|---:|---:|---:|
{rows}

F1采用三类宏平均；分类零分标签为中性，保留单独分类头。详细每类F1、加权F1及混淆矩阵见 metrics.json。训练曲线见 figures/training.png；缺失类型、比例与位置分析见 robustness.csv 与 figures/robustness.png。位置比较目前以文本模态50%缺失为代表，没有穷举全部组合。

验证集误差最大的20条样本见 validation_error_cases.csv，可按sample_id回查附件2原文。测试集中性F1为 {test['class_f1'][1]:.4f}，是当前版本较弱的类别；分类与回归双头可能在中性边界出现不一致。以上是对实际结果的归纳，不据测试集继续调参。

当前模型明显依赖文本：中段文本缺失70%时，验证MAE为 {text_missing.mae:.4f}、Macro F1为 {text_missing.macro_f1:.4f}；音频或视觉缺失的变化较小。这意味着已完成局部缺失预测流程，但还不能宣称模型对各种模态缺失同样稳健。

附件3的30条结果已导出 q2_predictions.csv。其为无标签专项集，不能计算专项准确率。门控旁路结果是推理阶段干预，不是重新训练的固定融合基线：旁路验证MAE={metrics['inference_gate_bypass']['mae']:.4f}，Macro F1={metrics['inference_gate_bypass']['macro_f1']:.4f}。

## 问题3

问题2与问题3共用已训练主干和参数，问题3增加模态遮挡、连续窗口遮挡及原视频定位。作用程度定义为遮挡前后固定预测类别概率/强度的有符号差值。主要参考模态按类别概率绝对变化选择；不会把门控权重称作因果贡献。

附件4的20条预测及解释见 q3_predictions_explanations.csv，逐窗口干预见 q3_window_effects.csv，解释卡见 cards。平均所选窗口绝对效应为 {fidelity.selected_window_abs_effect.mean():.4f}，随机同宽窗口为 {fidelity.random_window_abs_effect.mean():.4f}。所选窗口本来就是候选中的最大影响窗口，因此该比较仅用于复核，不是独立解释质量或因果有效性的证明。

原始特征没有时间戳。这里对原视频文本做CTC强制对齐，并在词元编号一致时按位置映射音视频证据。该映射仍假设原附件的对齐索引对应相同WordPiece，不能宣称恢复了官方精确时间戳。局部上下文编码也意味着词元门控不能等同于仅使用那一个原词；干预通过重新编码和重新运行门控衡量影响。

## 复现与范围

本次只运行一个随机种子和短训练，没有证明最优性。预训练BERT和ASR工具没有引入额外情感数据进行训练。问题1全部样本与标签保留；专项集未用于选参。所有数值以CSV和JSON为准。提交包不包含大体积通用预训练权重，需按照README下载并通过SHA256清单复核。

来源：Yang, Chen and Kim, Efficient Deep Visual and Inertial Odometry with Adaptive Visual Modality Selection, ECCV 2022, https://arxiv.org/abs/2205.06187 。代码参考 https://github.com/mingyuyng/Visual-Selective-VIO 。本赛题任务与数据口径以工作区E题题面为准。
'''
    (out / '结果说明.md').write_text(memo, encoding='utf-8')
    overview = ['<!doctype html><meta charset="utf-8"><title>E题结果总览</title>',
                '<style>body{font:16px system-ui;max-width:1100px;margin:30px auto;line-height:1.7}img{max-width:100%}td,th{padding:8px;border:1px solid #ddd}table{border-collapse:collapse}</style>',
                '<h1>E题结果总览</h1>', '<p>100条原始视频特征、30条缺失预测、20条解释预测。运行细节见结果说明.md。</p>',
                pd.DataFrame([{k:v for k,v in d.items() if k in ('mae','pearson','accuracy','macro_f1')} for d in (val,test)], index=['Validation','Test']).to_html(),
                '<img src="figures/training.png"><img src="figures/robustness.png"><img src="figures/confusion.png"><img src="figures/local_evidence.png">',
                '<h2>问题1典型对齐</h2><img src="q1/example_frame.jpg">', pd.DataFrame(example[:12]).to_html(index=False), '<h2>解释卡</h2>']
    for path in sorted((out / 'cards').glob('*.html')):
        overview.append(f'<a href="cards/{path.name}">{path.stem}</a>　')
    (out / '结果总览.html').write_text(''.join(overview), encoding='utf-8')
    package = ROOT / 'E题提交材料.zip'
    with zipfile.ZipFile(package, 'w', zipfile.ZIP_DEFLATED) as z:
        for pattern in ('*.py', '*.md', '*.txt', '*.ps1'):
            for path in ROOT.glob(pattern): z.write(path, path.name)
        for path in out.rglob('*'):
            if path.is_file() and path.suffix not in ('.log',):
                z.write(path, 'outputs/' + path.relative_to(out).as_posix())
        for name in ('bert', 'aligner'):
            manifest = ROOT / 'pretrained' / name / 'download_manifest.json'
            if manifest.exists(): z.write(manifest, f'pretrained/{name}/download_manifest.json')
    size = package.stat().st_size
    if size > 50_000_000: raise ValueError(f'Archive exceeds conservative 50 MB: {size}')
    save_json(out / 'delivery_check.json', {'q1_samples': 100, 'q2_predictions': 30, 'q3_explanations': 20,
                                          'q1_failed': failed, 'archive_bytes': size,
                                          'under_50MB': True})
    print(f'Exported {package.name}: {size / 1e6:.2f} MB', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', default=str(ROOT / 'outputs'))
    run(p.parse_args().output)
