"""Data contracts for the contest's aligned features (original files are read-only)."""
from pathlib import Path
import builtins
import json
import pickle
import random
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent / 'E题数据' / 'E题数据'
LABELS = ['Negative', 'Neutral', 'Positive']
MODALITIES = ['text', 'audio', 'vision']


class RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy', 'numpy.core.multiarray', 'numpy._core.multiarray',
                      'numpy.core.numeric', 'numpy._core.numeric') and name in (
                'ndarray', 'dtype', '_reconstruct', 'scalar', '_frombuffer', 'asarray'):
            return super().find_class(module, name)
        if module == 'builtins' and name in ('set', 'slice', 'complex'):
            return getattr(builtins, name)
        raise pickle.UnpicklingError(f'Unsupported pickle global: {module}.{name}')


def read_pickle(path):
    with open(path, 'rb') as stream:
        return RestrictedUnpickler(stream).load()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_record(record, fallback_id='sample'):
    """Return batches; time validity is separate from modality observation."""
    bert = np.asarray(record['text_bert'])
    if bert.ndim == 2:
        bert = bert[None]
    if bert.shape[1:] != (3, 50) or not np.isfinite(bert).all() or not np.equal(bert, np.round(bert)).all():
        raise ValueError('Expected integer-valued BERT triplets (N,3,50)')
    bert = bert.astype(np.int64)
    if bert[:, 0].min() < 0 or bert[:, 0].max() >= 30522:
        raise ValueError('Token IDs are outside bert-base-uncased vocabulary')
    if not np.isin(bert[:, 1], [0, 1]).all() or not np.isin(bert[:, 2], [0, 1]).all():
        raise ValueError('Invalid BERT attention/type mask')
    n = len(bert)
    audio = np.asarray(record['audio'], dtype=np.float32).reshape(n, 50, 74)
    vision = np.asarray(record['vision'], dtype=np.float32).reshape(n, 50, 35)
    finite_a, finite_v = np.isfinite(audio).all(-1), np.isfinite(vision).all(-1)
    audio, vision = np.nan_to_num(audio), np.nan_to_num(vision)
    ids, attn = bert[:, 0], bert[:, 1].astype(bool)
    special = (ids == 101) | (ids == 102)
    # Last observed token/frame bounds the real span. Internal text holes stay real.
    signal = attn | (np.abs(audio).sum(-1) > 0) | (np.abs(vision).sum(-1) > 0)
    last = np.where(signal, np.arange(50), -1).max(-1)
    valid = (np.arange(50)[None] <= last[:, None]) & ~special
    valid[:, 0] = False  # aligned files reserve initial CLS position
    availability = np.stack([
        attn & (ids != 0) & ~special,
        finite_a & (np.abs(audio).sum(-1) > 0),
        finite_v & (np.abs(vision).sum(-1) > 0)], -1) & valid[..., None]
    raw_ids = record.get('id', [fallback_id] * n)
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    raw_text = record.get('raw_text', [''] * n)
    if isinstance(raw_text, (str, np.str_)):
        raw_text = [str(raw_text)]
    result = dict(bert=bert, audio=audio, vision=vision, valid=valid,
                  availability=availability, ids=[str(x) for x in raw_ids],
                  raw_text=[str(x) for x in raw_text])
    if 'regression_labels' in record:
        y = np.asarray(record['regression_labels'], dtype=np.float32).reshape(n)
        if not np.isfinite(y).all() or (np.abs(y) > 3.001).any():
            raise ValueError('Invalid sentiment labels')
        result['y'] = y
        result['classes'] = np.where(y < 0, 0, np.where(y > 0, 2, 1)).astype(np.int64)
    return result


def load_splits(data=DATA):
    path = next(Path(data).rglob('aligned_50.pkl'))
    raw = read_pickle(path)
    return {split: normalize_record(raw[split]) for split in ('train', 'valid', 'test')}


def special_records(attachment, data=DATA):
    paths = sorted(p for p in Path(data).rglob('*.pkl')
                   if f'附件{attachment}' in str(p) and '未对齐' not in str(p))
    for path in paths:
        raw = read_pickle(path)
        if 'test' in raw:
            raw = raw['test']
        out = normalize_record(raw, path.stem)
        yield path, out


def batch_from(data, indices, device):
    result = {k: torch.as_tensor(data[k][indices], device=device)
              for k in ('bert', 'audio', 'vision', 'valid', 'availability')}
    for k in ('y', 'classes'):
        if k in data:
            result[k] = torch.as_tensor(data[k][indices], device=device)
    return result


def mask_batch(batch, drop):
    """drop is (B,T,3), applied BEFORE contextual text encoding."""
    out = {k: v.clone() for k, v in batch.items()}
    out['availability'] &= ~drop
    text_drop = drop[:, :, 0]
    out['bert'][:, 0].masked_fill_(text_drop, 0)
    out['bert'][:, 1].masked_fill_(text_drop, 0)
    out['audio'].masked_fill_(drop[:, :, 1, None], 0)
    out['vision'].masked_fill_(drop[:, :, 2, None], 0)
    return out


def block_drop(batch, rate=None, modality=None, position=None, probability=0.65, generator=None):
    """Mask real contiguous index spans. Return known synthetic missingness."""
    drop = torch.zeros_like(batch['availability'])
    rng = generator if generator is not None else np.random.default_rng()
    for b in range(len(drop)):
        if rate is None and rng.random() > probability:
            continue
        real = batch['valid'][b].nonzero().flatten().cpu().numpy()
        if len(real) == 0:
            continue
        frac = float(rate) if rate is not None else rng.uniform(0.1, 0.6)
        if frac <= 0:
            continue
        width = max(1, int(np.ceil(len(real) * frac)))
        width = min(width, len(real))
        if position == 'front': start = 0
        elif position == 'middle': start = (len(real) - width) // 2
        elif position == 'back': start = len(real) - width
        else: start = int(rng.integers(len(real) - width + 1))
        mods = [modality] if modality is not None else rng.choice(3, int(rng.integers(1, 4)), replace=False)
        loc = real[start:start + width]
        for m in mods:
            drop[b, int(loc[0]):int(loc[-1]) + 1, int(m)] = True
    return mask_batch(batch, drop), drop


def fit_stats(train):
    stats = {}
    for j, name in enumerate(('audio', 'vision'), 1):
        x = train[name][train['availability'][:, :, j]]
        mean = x.mean(0) if len(x) else np.zeros(train[name].shape[-1])
        std = x.std(0) if len(x) else np.ones_like(mean)
        stats[name] = {'mean': mean.tolist(), 'std': np.maximum(std, 1e-4).tolist()}
    return stats

