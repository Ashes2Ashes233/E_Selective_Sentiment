import argparse
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from transformers import BertTokenizerFast
from common import (ROOT, DATA, LABELS, MODALITIES, special_records, batch_from, mask_batch, save_json)
from train import load_checkpoint, text_features
from media import EvidenceAligner, read_audio, align_tokens


@torch.no_grad()
def infer(model, encoder, batch, device):
    output = model(text_features(encoder, batch['bert'], device), batch)
    return {'prob': output['logits'].softmax(-1).cpu().numpy(),
            'score': output['score'].cpu().numpy(), 'gates': output['gates'].cpu().numpy()}


def repeated(batch, count):
    return {k: v.repeat((count,) + (1,) * (v.ndim - 1)) for k, v in batch.items()}


def base_row(sample_id, output):
    p = output['prob'][0]
    return {'sample_id': sample_id, 'predicted_class': LABELS[int(p.argmax())],
            'predicted_score': float(output['score'][0]),
            **{f'prob_{label}': float(p[i]) for i, label in enumerate(LABELS)}}


def save_frame(video, seconds, path):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000 * seconds)
    ok, frame = cap.read()
    cap.release()
    if ok:
        if frame.shape[1] > 640:
            frame = cv2.resize(frame, (640, int(frame.shape[0] * 640 / frame.shape[1])))
        success, buffer = cv2.imencode('.jpg', frame)
        if success:
            path.parent.mkdir(parents=True, exist_ok=True)
            buffer.tofile(str(path))
            return True
    return False


def explain(model, encoder, batch, device, window=4):
    original = infer(model, encoder, batch, device)
    target = int(original['prob'][0].argmax())
    real = batch['valid'][0].nonzero().flatten().cpu().numpy()
    masks, descriptions = [], []
    for m in range(3):
        mask = torch.zeros_like(batch['availability'])
        mask[:, :, m] = batch['valid']
        masks.append(mask[0]); descriptions.append(('modality', m, -1, -1))
    if len(real):
        for m in range(3):
            for start in range(int(real[0]), int(real[-1]) + 1, window):
                end = min(start + window, int(real[-1]) + 1)
                mask = torch.zeros_like(batch['availability'])
                mask[:, start:end, m] = True
                masks.append(mask[0]); descriptions.append(('window', m, start, end))
    variants = repeated(batch, len(masks))
    variants = mask_batch(variants, torch.stack(masks))
    all_prob, all_score = [], []
    for offset in range(0, len(masks), 32):
        sliced = {k: v[offset:offset + 32] for k, v in variants.items()}
        pred = infer(model, encoder, sliced, device)
        all_prob.extend(pred['prob']); all_score.extend(pred['score'])
    rows = []
    for desc, prob, score in zip(descriptions, all_prob, all_score):
        rows.append({'kind': desc[0], 'modality': MODALITIES[desc[1]], 'start_index': desc[2],
                     'end_index_exclusive': desc[3],
                     'delta_target_probability': float(original['prob'][0, target] - prob[target]),
                     'delta_score': float(original['score'][0] - score)})
    return original, rows


def card_html(row, evidence, tokens, gates, output):
    text = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>情感解释卡</title>',
            '<style>body{font:16px system-ui;max-width:1000px;margin:35px auto;line-height:1.7}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:7px}.token{display:inline-block;padding:4px;margin:2px}img{max-width:640px}</style>',
            f'<h1>样本 {html.escape(row["sample_id"])} 情感解释</h1>',
            f'<p>预测：{row["predicted_class"]}；强度：{row["predicted_score"]:.3f}；主要参考模态：{row["primary_modality"]}</p>',
            '<p>下表贡献为遮挡前后固定目标类别概率的差值。正值为支持，负值为抑制；不是现实因果贡献或可加百分比。</p>',
            '<table><tr><th>模态</th><th>类别概率变化</th><th>强度变化</th><th>关键证据</th></tr>']
    for ev in evidence:
        text.append(f'<tr><td>{ev["modality"]}</td><td>{ev["modality_delta_probability"]:.4f}</td>'
                    f'<td>{ev["modality_delta_score"]:.4f}</td><td>{html.escape(ev.get("excerpt", ""))} '
                    f'{ev.get("start_seconds")}–{ev.get("end_seconds")}秒 ({ev.get("timing_quality")})</td></tr>')
    text.append('</table><h2>三模态选择强度</h2><p>颜色越深表示门控越大；此图只显示选择行为。</p>')
    for m, name in enumerate(MODALITIES):
        text.append(f'<p>{name}</p>')
        for t, token in enumerate(tokens):
            if token in ('[PAD]', '[CLS]', '[SEP]'): continue
            alpha = float(gates[t, m]) * .7
            text.append(f'<span class="token" style="background:rgba(50,130,200,{alpha:.3f})" title="位置{t}，权重{gates[t,m]:.3f}">{html.escape(token)}</span>')
    text.append('<h2>局部窗口遮挡影响</h2>')
    text.append('<p>关键窗口按对固定目标类别的绝对影响选择，可能提供支持或抑制证据。完整逐窗口数值见配套CSV。</p>')
    text.append(f'<img src="../keyframes/{html.escape(row["file_id"])}.jpg" alt="视觉证据关键帧">')
    output.write_text(''.join(text), encoding='utf-8')


def run(args):
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, encoder, _ = load_checkpoint(args.checkpoint, device)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    q2 = []
    for path, data in special_records(3, args.data):
        batch = batch_from(data, slice(None), device)
        pred = infer(model, encoder, batch, device)
        row = base_row(path.stem, pred)
        row['input_file'] = path.name
        for m, name in enumerate(MODALITIES):
            row[f'{name}_observed_fraction'] = float((batch['availability'][0, :, m].sum() / batch['valid'][0].sum().clamp_min(1)).cpu())
        q2.append(row)
    pd.DataFrame(q2).to_csv(out / 'q2_predictions.csv', index=False, encoding='utf-8-sig')
    print(f'Q2 exported {len(q2)} samples', flush=True)
    tokenizer = BertTokenizerFast.from_pretrained(str(ROOT / 'pretrained' / 'bert'), local_files_only=True)
    aligner = EvidenceAligner(device)
    q3, all_effects, fidelity = [], [], []
    for sub in ('cards', 'keyframes', 'q3_alignment'): (out / sub).mkdir(exist_ok=True)
    for path, data in special_records(4, args.data):
        batch = batch_from(data, slice(None), device)
        pred, effects = explain(model, encoder, batch, device, args.window)
        row = base_row(data['ids'][0], pred)
        row['file_id'] = path.stem
        mods = [e for e in effects if e['kind'] == 'modality']
        row['primary_modality'] = max(mods, key=lambda e: abs(e['delta_target_probability']))['modality']
        video = path.parent / 'videos' / f'{path.stem}.mp4'
        mapping = align_tokens(data['raw_text'][0], read_audio(video), tokenizer, aligner)
        # Do not claim exact source localization if token sequences differ.
        token_match = np.array_equal(mapping['input_ids'], data['bert'][0, 0])
        mapping.update(sample_id=data['ids'][0], token_ids_match=bool(token_match))
        save_json(out / 'q3_alignment' / f'{path.stem}.json', mapping)
        tokens = tokenizer.convert_ids_to_tokens(data['bert'][0, 0].tolist())
        evidence = []
        for m, name in enumerate(MODALITIES):
            candidates = [e for e in effects if e['kind'] == 'window' and e['modality'] == name]
            best = max(candidates, key=lambda e: abs(e['delta_target_probability'])) if candidates else None
            mod = mods[m]
            row[f'{name}_delta_probability'] = mod['delta_target_probability']
            row[f'{name}_delta_score'] = mod['delta_score']
            ev = {'modality': name, 'modality_delta_probability': mod['delta_target_probability'],
                  'modality_delta_score': mod['delta_score']}
            if best:
                a, b = best['start_index'], best['end_index_exclusive']
                spans = [s for s in mapping['token_spans'][a:b] if s[1] > s[0]]
                ev.update(best, excerpt=tokenizer.convert_tokens_to_string(tokens[a:b]),
                          start_seconds=min(s[0] for s in spans) if spans and token_match else None,
                          end_seconds=max(s[1] for s in spans) if spans and token_match else None,
                          timing_quality=(mapping['status'] + '_feature_position_mapping_assumed') if token_match else 'unresolved_token_mismatch')
                if name == 'vision' and ev['start_seconds'] is not None:
                    save_frame(video, (ev['start_seconds'] + ev['end_seconds']) / 2, out / 'keyframes' / f'{path.stem}.jpg')
                # Same-width random window comparison: faithfulness proxy, not label accuracy.
                real = batch['valid'][0].nonzero().flatten().cpu().numpy()
                width = b - a
                start = int(np.random.default_rng(2026 + int(path.stem) * 3 + m).choice(real))
                start = min(start, max(int(real[0]), int(real[-1]) + 1 - width))
                drop = torch.zeros_like(batch['availability']); drop[:, start:start + width, m] = True
                random_pred = infer(model, encoder, mask_batch(batch, drop), device)
                target = int(pred['prob'][0].argmax())
                fidelity.append({'sample_id': row['sample_id'], 'modality': name,
                                 'selected_window_abs_effect': abs(best['delta_target_probability']),
                                 'random_window_abs_effect': abs(float(pred['prob'][0, target] - random_pred['prob'][0, target]))})
            evidence.append(ev)
        row['evidence_json'] = json.dumps(evidence, ensure_ascii=False)
        row['alignment_status'] = mapping['status']
        row['position_mapping_note'] = 'WordPiece-aligned feature positions assumed; original extraction timestamps unavailable'
        q3.append(row)
        for effect in effects: all_effects.append({'sample_id': row['sample_id'], **effect})
        card_html(row, evidence, tokens, pred['gates'][0], out / 'cards' / f'{path.stem}.html')
        print(f'Q3 {len(q3)}/20 {path.stem}: {row["predicted_class"]}', flush=True)
    pd.DataFrame(q3).to_csv(out / 'q3_predictions_explanations.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(all_effects).to_csv(out / 'q3_window_effects.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(fidelity).to_csv(out / 'explanation_fidelity.csv', index=False, encoding='utf-8-sig')
    save_json(out / 'submission_counts.json', {'q2': len(q2), 'q3': len(q3)})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(DATA))
    p.add_argument('--checkpoint', default=str(ROOT / 'outputs' / 'best.pt'))
    p.add_argument('--output', default=str(ROOT / 'outputs'))
    p.add_argument('--window', type=int, default=4)
    run(p.parse_args())
