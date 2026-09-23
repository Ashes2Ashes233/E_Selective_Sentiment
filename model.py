"""Adapted from Yang et al., ECCV 2022: policy -> gated fusion -> recurrent state.

Task-specific encoders, masks, sequence pooling and heads replace VIO components.
Frozen BERT is deliberately outside saved task parameters.
"""
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


class FrozenText(nn.Module):
    def __init__(self, model_dir):
        super().__init__()
        self.bert = BertModel.from_pretrained(str(model_dir), local_files_only=True)
        self.bert.requires_grad_(False)
        self.bert.eval()

    def forward(self, triplets):
        self.bert.eval()
        ids, mask, types = triplets[:, 0], triplets[:, 1], triplets[:, 2]
        # Prevent entirely masked attention rows; CLS remains a non-evidence token.
        empty = mask.sum(-1) == 0
        if empty.any():
            ids, mask = ids.clone(), mask.clone()
            ids[empty, 0], mask[empty, 0] = 101, 1
        with torch.no_grad():
            return self.bert(input_ids=ids, attention_mask=mask,
                             token_type_ids=types).last_hidden_state


class SelectiveSentiment(nn.Module):
    def __init__(self, stats, dim=96, hidden=192, gate='soft', dropout=0.2):
        super().__init__()
        self.config = dict(dim=dim, hidden=hidden, gate=gate, dropout=dropout)
        self.gate_mode = gate
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Linear(d, dim), nn.LayerNorm(dim), nn.GELU())
            for d in (768, 74, 35)])
        for name in ('audio', 'vision'):
            self.register_buffer(name + '_mean', torch.tensor(stats[name]['mean']))
            self.register_buffer(name + '_std', torch.tensor(stats[name]['std']))
        self.policy = nn.Sequential(nn.Linear(3 * dim + hidden + 3, 128),
                                    nn.LayerNorm(128), nn.GELU(), nn.Linear(128, 6))
        self.rnn = nn.GRUCell(3 * dim + 3, hidden)
        self.attention = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 3)
        self.regressor = nn.Linear(hidden, 1)

    def forward(self, text, batch, temperature=1.0):
        avail = batch['availability'] & batch['valid'][..., None]
        audio = ((batch['audio'] - self.audio_mean) / self.audio_std).clamp(-8, 8)
        vision = ((batch['vision'] - self.vision_mean) / self.vision_std).clamp(-8, 8)
        z = torch.stack([p(x) for p, x in zip(self.projections, (text, audio, vision))], 2)
        z = z * avail[..., None]
        h = z.new_zeros(len(z), self.config['hidden'])
        states, gates = [], []
        for t in range(z.shape[1]):
            a = avail[:, t].float()
            current = z[:, t].flatten(1)
            logits = self.policy(torch.cat([current, h, a], -1)).view(-1, 3, 2)
            if self.gate_mode == 'fixed':
                gate = torch.ones_like(a)
            elif self.gate_mode == 'hard':
                if self.training:
                    gate = F.gumbel_softmax(logits, tau=temperature, hard=True, dim=-1)[..., 1]
                else:
                    gate = (logits[..., 1] >= logits[..., 0]).float()
                scores = logits[..., 1] - logits[..., 0]
                fallback = F.one_hot(scores.masked_fill(~avail[:, t], -1e9).argmax(-1), 3).float()
                all_closed = ((gate * a).sum(-1) == 0) & (a.sum(-1) > 0)
                gate = torch.where(all_closed[:, None], fallback, gate)
            else:
                gate = (logits[..., 1] - logits[..., 0]).sigmoid()
            gate = gate * a
            fused = torch.cat([(z[:, t] * gate[..., None]).flatten(1), a], -1)
            proposed = self.rnn(fused, h)
            # Missing real positions can update on missingness; padding cannot.
            h = torch.where(batch['valid'][:, t, None], proposed, h)
            states.append(h)
            gates.append(gate)
        states = torch.stack(states, 1)
        valid = batch['valid'].clone()
        valid[~valid.any(-1), 0] = True
        pool = self.attention(states).squeeze(-1).masked_fill(~valid, -1e4).softmax(-1)
        summary = self.dropout((states * pool[..., None]).sum(1))
        return {'logits': self.classifier(summary),
                'score': 3 * torch.tanh(self.regressor(summary).squeeze(-1)),
                'gates': torch.stack(gates, 1), 'pool': pool}

