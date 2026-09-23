"""Small correctness checks, not an experimental benchmark."""
import numpy as np
import torch
from common import load_splits, fit_stats, batch_from, mask_batch, ROOT
from model import SelectiveSentiment, FrozenText
from train import text_features

torch.set_num_threads(4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = load_splits()
model = SelectiveSentiment(fit_stats(data['train'])).to(device)
encoder = FrozenText(ROOT / 'pretrained' / 'bert').to(device)
batch = batch_from(data['train'], slice(0, 4), device)
drop = torch.zeros_like(batch['availability']); drop[:, 2:6, 0] = True
masked = mask_batch(batch, drop)
assert not masked['bert'][:, 1, 2:6].any(), 'Text still observable after masking'
text = text_features(encoder, masked['bert'], device)
model.train()
out = model(text, masked)
loss = (out['score'] - batch['y']).square().mean() + torch.nn.functional.cross_entropy(out['logits'], batch['classes'])
loss.backward()
assert torch.isfinite(loss) and model.policy[-1].weight.grad.abs().sum() > 0
model.eval()
with torch.no_grad():
    a = model(text, masked)
    b = model(text, masked)
    assert torch.equal(a['score'], b['score']), 'Inference should be deterministic'
    assert not a['gates'][~masked['availability']].any(), 'Missing modality selected'
    padding_change = {k: v.clone() for k, v in masked.items()}
    padding_change['audio'][~masked['valid']] = 500
    padding_change['vision'][~masked['valid']] = -500
    corrupted_text = text.clone(); corrupted_text[~masked['valid']] = 500
    changed = model(corrupted_text, padding_change)
    assert torch.allclose(a['score'], changed['score'], atol=1e-6), 'Padding affects prediction'
    empty = mask_batch(batch, torch.ones_like(batch['availability']))
    empty_output = model(text_features(encoder, empty['bert'], device), empty)
    assert torch.isfinite(empty_output['score']).all(), 'All-missing input is not handled'
print('PASS: real data forward/backward, policy gradients, text masking, missing gates, deterministic inference, padding invariance, all-missing input', flush=True)

