import argparse
import json
import time
import platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from common import (ROOT, DATA, LABELS, MODALITIES, save_json, seed_all, load_splits,
                    batch_from, block_drop, fit_stats)
from model import FrozenText, SelectiveSentiment


def metrics(y, classes, scores, probabilities):
    pred = probabilities.argmax(-1)
    pearson = float(np.corrcoef(y, scores)[0, 1]) if np.std(y) > 1e-9 and np.std(scores) > 1e-9 else 0.0
    return {'n': len(y), 'mae': float(np.abs(y - scores).mean()), 'pearson': pearson,
            'accuracy': float(accuracy_score(classes, pred)),
            'macro_f1': float(f1_score(classes, pred, labels=[0, 1, 2], average='macro', zero_division=0)),
            'weighted_f1': float(f1_score(classes, pred, labels=[0, 1, 2], average='weighted', zero_division=0)),
            'class_f1': f1_score(classes, pred, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
            'confusion_matrix': confusion_matrix(classes, pred, labels=[0, 1, 2]).tolist()}


def text_features(encoder, triplets, device):
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
        return encoder(triplets).float()


@torch.no_grad()
def evaluate(model, encoder, data, device, batch_size=48, corruption=None):
    model.eval()
    scores, probabilities, usage = [], [], []
    rng = np.random.default_rng(2026)
    for start in range(0, len(data['bert']), batch_size):
        batch = batch_from(data, slice(start, start + batch_size), device)
        if corruption:
            batch, _ = block_drop(batch, generator=rng, **corruption)
        out = model(text_features(encoder, batch['bert'], device), batch)
        scores.extend(out['score'].cpu().numpy())
        probabilities.extend(out['logits'].softmax(-1).cpu().numpy())
        usage.extend((out['gates'].sum(1) / batch['valid'].sum(1).clamp_min(1)[:, None]).cpu().numpy())
    scores, probabilities = np.asarray(scores), np.asarray(probabilities)
    result = metrics(data['y'], data['classes'], scores, probabilities)
    result['mean_gate_per_valid_position'] = np.mean(usage, axis=0).tolist()
    return result, scores, probabilities


def prediction_frame(data, scores, probabilities):
    df = pd.DataFrame({'sample_id': data['ids'], 'true_score': data['y'],
                       'true_class': [LABELS[i] for i in data['classes']],
                       'predicted_score': scores,
                       'predicted_class': [LABELS[i] for i in probabilities.argmax(-1)]})
    for i, label in enumerate(LABELS): df['prob_' + label] = probabilities[:, i]
    return df


def load_checkpoint(path, device, bert_dir=None):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    model = SelectiveSentiment(saved['stats'], **saved['model_config']).to(device)
    model.load_state_dict(saved['state_dict'])
    encoder = FrozenText(bert_dir or ROOT / 'pretrained' / 'bert').to(device)
    model.eval()
    return model, encoder, saved


def run(args):
    warmup_epochs = args.warmup_epochs
    joint_epochs = args.joint_epochs if args.joint_epochs is not None else args.epochs - warmup_epochs
    if warmup_epochs < 0 or joint_epochs <= 0 or args.patience < 1:
        raise ValueError('warmup-epochs must be >= 0; joint epochs and patience must be positive')
    total_epochs = warmup_epochs + joint_epochs
    seed_all(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(outdir / 'training_config.json', {k: v for k, v in vars(args).items()
                                               if k not in ('data', 'bert', 'output')})
    splits = load_splits(args.data)
    # A short tokenizer check identifies the supplied IDs without using held-out labels.
    from transformers import BertTokenizerFast
    tokenizer = BertTokenizerFast.from_pretrained(args.bert, local_files_only=True)
    checks = []
    for i in range(min(40, len(splits['train']['bert']))):
        ids = tokenizer(splits['train']['raw_text'][i], max_length=50, truncation=True, padding='max_length')['input_ids']
        checks.append(bool(np.array_equal(ids, splits['train']['bert'][i, 0])))
    save_json(outdir / 'input_audit.json', {'splits': {k: len(v['bert']) for k, v in splits.items()},
                'tokenizer_exact_match_checked': len(checks), 'tokenizer_exact_matches': sum(checks),
                'label_order': LABELS, 'text_path': 'frozen full BERT; corruption before encoding',
                'source': 'contest aligned_50.pkl; no external sentiment data'})
    if sum(checks) < 0.9 * len(checks):
        raise ValueError('BERT tokenizer does not match supplied input IDs. See input_audit.json')
    stats = fit_stats(splits['train'])
    model = SelectiveSentiment(stats, args.dim, args.hidden, args.gate).to(device)
    encoder = FrozenText(args.bert).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    counts = np.bincount(splits['train']['classes'], minlength=3)
    weights = torch.tensor(np.sqrt(counts.sum() / (3 * counts)), device=device, dtype=torch.float32)
    # A nonzero warmup_epochs enables the strict two-stage schedule:
    # complete observations with fixed fusion first, followed by joint
    # missing-modality and policy training.  With warmup_epochs=0 the
    # original single-stage behavior is preserved.
    history, best, stale = [], float('inf'), 0
    rng = np.random.default_rng(args.seed)
    print(f'device={device}, train={sum(counts)}, task_parameters={sum(p.numel() for p in model.parameters()):,}', flush=True)
    for epoch in range(1, total_epochs + 1):
        is_warmup = warmup_epochs > 0 and epoch <= warmup_epochs
        stage = 'warmup' if is_warmup else ('joint' if warmup_epochs else 'single')
        # During warmup the policy is bypassed so the task network learns
        # from complete observations.  The policy starts learning only in
        # the joint stage, where synthetic contiguous missingness is added.
        model.gate_mode = 'fixed' if is_warmup else args.gate
        model.policy.requires_grad_(not is_warmup)
        model.train()
        started, loss_sum, seen = time.time(), 0., 0
        indices = rng.permutation(len(splits['train']['bert']))
        for start in range(0, len(indices), args.batch_size):
            batch = batch_from(splits['train'], indices[start:start + args.batch_size], device)
            if not is_warmup:
                probability = 0.25 if warmup_epochs == 0 and epoch == 1 else 0.65
                batch, _ = block_drop(batch, probability=probability, generator=rng)
            out = model(text_features(encoder, batch['bert'], device), batch,
                        temperature=max(0.5, 2.0 * 0.9 ** max(1, epoch - warmup_epochs)))
            loss = F.smooth_l1_loss(out['score'], batch['y'], beta=0.5)
            loss = loss + 0.7 * F.cross_entropy(out['logits'], batch['classes'], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch['bert'])
            seen += len(batch['bert'])
        clean, _, _ = evaluate(model, encoder, splits['valid'], device, args.batch_size)
        missing, _, _ = evaluate(model, encoder, splits['valid'], device, args.batch_size,
                                  {'rate': 0.4})
        criterion = 0.5 * (clean['mae'] + missing['mae']) + 0.15 * (2 - clean['macro_f1'] - missing['macro_f1'])
        row = {'epoch': epoch, 'stage': stage, 'loss': loss_sum / seen, 'validation_mae': clean['mae'],
               'validation_macro_f1': clean['macro_f1'], 'missing_mae': missing['mae'],
               'criterion': criterion, 'seconds': time.time() - started}
        history.append(row)
        pd.DataFrame(history).to_csv(outdir / 'training_history.csv', index=False)
        print(json.dumps(row), flush=True)
        if is_warmup:
            # Keep an explicit warmup checkpoint for inspection, but do not
            # select it as best.pt because its policy is intentionally bypassed.
            torch.save({'state_dict': model.state_dict(), 'stats': stats,
                        'model_config': dict(model.config, gate='fixed'),
                        'epoch': epoch, 'seed': args.seed, 'criterion': criterion,
                        'stage': stage, 'labels': LABELS}, outdir / 'warmup.pt')
        else:
            if criterion < best:
                best, stale = criterion, 0
                torch.save({'state_dict': model.state_dict(), 'stats': stats, 'model_config': model.config,
                            'epoch': epoch, 'seed': args.seed, 'criterion': criterion,
                            'stage': stage, 'labels': LABELS}, outdir / 'best.pt')
            else:
                stale += 1
            if stale >= args.patience:
                print(f'Early stop at epoch {epoch}: {stale} epochs without validation improvement.', flush=True)
                break
    saved = torch.load(outdir / 'best.pt', map_location=device, weights_only=True)
    model.load_state_dict(saved['state_dict'])
    model.gate_mode = args.gate
    report = {}
    for split in ('valid', 'test'):
        result, scores, probs = evaluate(model, encoder, splits[split], device, args.batch_size)
        report[split] = result
        prediction_frame(splits[split], scores, probs).to_csv(outdir / f'{split}_predictions.csv', index=False, encoding='utf-8-sig')
    robustness = []
    for m in range(3):
        for rate in (0.2, 0.5, 0.7):
            result, _, _ = evaluate(model, encoder, splits['valid'], device, args.batch_size,
                                    {'rate': rate, 'modality': m, 'position': 'middle'})
            robustness.append({'modality': MODALITIES[m], 'rate': rate, 'position': 'middle',
                               **{k: result[k] for k in ('mae', 'pearson', 'accuracy', 'macro_f1')}})
    for position in ('front', 'back'):
        result, _, _ = evaluate(model, encoder, splits['valid'], device, args.batch_size,
                                {'rate': 0.5, 'modality': 0, 'position': position})
        robustness.append({'modality': 'text', 'rate': 0.5, 'position': position,
                           **{k: result[k] for k in ('mae', 'pearson', 'accuracy', 'macro_f1')}})
    pd.DataFrame(robustness).to_csv(outdir / 'robustness.csv', index=False)
    original_mode = model.gate_mode
    model.gate_mode = 'fixed'
    intervention, _, _ = evaluate(model, encoder, splits['valid'], device, args.batch_size)
    model.gate_mode = original_mode
    report['inference_gate_bypass'] = intervention
    report['ablation_note'] = 'Inference intervention only; not an independently retrained fixed-fusion baseline.'
    report['best_epoch'] = saved['epoch']
    report['best_stage'] = saved.get('stage', 'single')
    report['schedule'] = {'warmup_epochs': warmup_epochs, 'joint_epochs': joint_epochs,
                          'total_epochs_requested': total_epochs}
    report['runtime'] = {'python': platform.python_version(), 'torch': torch.__version__, 'device': str(device)}
    save_json(outdir / 'metrics.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    from pathlib import Path
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(DATA))
    p.add_argument('--bert', default=str(ROOT / 'pretrained' / 'bert'))
    p.add_argument('--output', default=str(ROOT / 'outputs'))
    p.add_argument('--epochs', type=int, default=8)
    p.add_argument('--warmup-epochs', type=int, default=0,
                   help='Strict complete-modality warmup epochs; 0 keeps single-stage training')
    p.add_argument('--joint-epochs', type=int, default=None,
                   help='Joint missing-modality/policy epochs; defaults to epochs-warmup-epochs')
    p.add_argument('--patience', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--dim', type=int, default=96)
    p.add_argument('--hidden', type=int, default=192)
    p.add_argument('--lr', type=float, default=0.001)
    p.add_argument('--gate', choices=['soft', 'hard', 'fixed'], default='soft')
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--cpu', action='store_true')
    run(p.parse_args())
