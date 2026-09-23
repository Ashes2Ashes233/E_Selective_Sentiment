"""Q1 raw features and Q3 evidence timing. No sentiment labels are used here."""
import argparse
import json
import re
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
import librosa
import imageio_ffmpeg
from transformers import BertTokenizerFast, Wav2Vec2Processor, Wav2Vec2ForCTC
from common import ROOT, DATA, save_json
from model import FrozenText
from train import text_features


def read_audio(path):
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    process = subprocess.run([ffmpeg, '-v', 'error', '-i', str(path), '-f', 'f32le',
                              '-ac', '1', '-ar', '16000', '-'], capture_output=True, check=True)
    return np.frombuffer(process.stdout, dtype='<f4').copy()


def ctc_path(logp, target, blank):
    """Exact CTC Viterbi over blank-expanded transcript; O(frames * characters)."""
    target = np.asarray(target, dtype=np.int64)
    ext = np.full(2 * len(target) + 1, blank, dtype=np.int64)
    ext[1::2] = target
    tmax, states = len(logp), len(ext)
    if not len(target) or tmax < len(target): raise ValueError('Transcript cannot fit audio frames')
    score = np.full(states, -np.inf, dtype=np.float32)
    score[0], score[1] = logp[0, blank], logp[0, target[0]]
    back = np.zeros((tmax, states), dtype=np.uint8)
    can_skip = np.zeros(states, dtype=bool)
    can_skip[2:] = (ext[2:] != blank) & (ext[2:] != ext[:-2])
    for t in range(1, tmax):
        one = np.r_[-np.inf, score[:-1]]
        two = np.r_[-np.inf, -np.inf, score[:-2]]
        two[~can_skip] = -np.inf
        candidates = np.stack([score, one, two])
        choice = candidates.argmax(0)
        score = candidates[choice, np.arange(states)] + logp[t, ext]
        back[t] = choice
    state = states - 1 if score[-1] >= score[-2] else states - 2
    if not np.isfinite(score[state]): raise ValueError('No valid CTC alignment')
    path = np.zeros(tmax, dtype=np.int32)
    for t in range(tmax - 1, -1, -1):
        path[t] = state
        if t: state -= int(back[t, state])
    return path


class EvidenceAligner:
    def __init__(self, device, model_dir=None):
        self.device = device
        model_dir = model_dir or ROOT / 'pretrained' / 'aligner'
        self.processor = Wav2Vec2Processor.from_pretrained(str(model_dir), local_files_only=True)
        self.model = Wav2Vec2ForCTC.from_pretrained(str(model_dir), local_files_only=True).to(device).eval()
        self.vocab = self.processor.tokenizer.get_vocab()

    @torch.no_grad()
    def words(self, waveform, text):
        matches = list(re.finditer(r"[A-Za-z]+(?:'[A-Za-z]+)?", text))
        pieces, target, word_ranges = [], [], []
        for i, match in enumerate(matches):
            if i: target.append(self.vocab['|'])
            start = len(target)
            token = match.group().upper()
            target.extend(self.vocab[ch] for ch in token if ch in self.vocab)
            word_ranges.append((start, len(target)))
            pieces.append({'word': match.group(), 'char_start': match.start(), 'char_end': match.end()})
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors='pt')
        values = inputs.input_values.to(self.device)
        # Float32 keeps CTC confidence and alignment numerically stable.
        logp = self.model(values).logits[0].log_softmax(-1).cpu().numpy()
        path = ctc_path(logp, target, self.model.config.pad_token_id)
        seconds = len(waveform) / 16000 / len(logp)
        for info, (left, right) in zip(pieces, word_ranges):
            frames = np.where((path >= 2 * left + 1) & (path <= 2 * (right - 1) + 1))[0]
            if not len(frames): raise ValueError('Word has no aligned frames')
            emission_frames = frames[path[frames] % 2 == 1]
            chars = (path[emission_frames] - 1) // 2
            conf = np.exp(logp[emission_frames, np.asarray(target)[chars]]).mean()
            info.update(start=float(frames[0] * seconds), end=float((frames[-1] + 1) * seconds),
                        confidence=float(conf))
        return pieces


def align_tokens(text, waveform, tokenizer, aligner):
    encoded = tokenizer(text, truncation=True, max_length=50, padding='max_length',
                        return_offsets_mapping=True)
    duration = len(waveform) / 16000
    status, error = 'ctc_forced_alignment', ''
    try:
        words = aligner.words(waveform, text)
    except Exception as exc:
        # Preserve every sample and openly label a coarse fallback.
        status, error = 'approximate_uniform_word_fallback', str(exc)
        matches = list(re.finditer(r'\S+', text))
        words = [{'word': m.group(), 'char_start': m.start(), 'char_end': m.end(),
                  'start': duration * i / max(len(matches), 1),
                  'end': duration * (i + 1) / max(len(matches), 1), 'confidence': 0.0}
                 for i, m in enumerate(matches)]
    spans, quality = [], []
    for start, end in encoded['offset_mapping']:
        overlaps = [w for w in words if w['char_start'] < end and w['char_end'] > start]
        if end <= start:
            spans.append([0., 0.]); quality.append('special_or_padding')
        elif overlaps:
            spans.append([min(w['start'] for w in overlaps), max(w['end'] for w in overlaps)])
            quality.append(status)
        else:
            # Punctuation/numerals have no exact CTC word; locate near adjacent word.
            nearest = min(words, key=lambda w: abs(w['char_start'] - start)) if words else None
            spans.append([nearest['start'], nearest['end']] if nearest else [0., duration])
            quality.append('approximate_nearest_word')
    return {'words': words, 'token_spans': spans, 'token_quality': quality,
            'token_offsets': encoded['offset_mapping'], 'input_ids': encoded['input_ids'],
            'attention_mask': encoded['attention_mask'], 'status': status, 'error': error,
            'duration': duration}


def audio_features(waveform):
    hop = 256
    mfcc = librosa.feature.mfcc(y=waveform, sr=16000, n_mfcc=13, n_fft=512, hop_length=hop)
    width = min(9, mfcc.shape[1] if mfcc.shape[1] % 2 else mfcc.shape[1] - 1)
    delta = librosa.feature.delta(mfcc, width=max(3, width), mode='nearest')
    specs = [librosa.feature.rms(y=waveform, frame_length=512, hop_length=hop),
             librosa.feature.zero_crossing_rate(waveform, frame_length=512, hop_length=hop),
             librosa.feature.spectral_centroid(y=waveform, sr=16000, n_fft=512, hop_length=hop),
             librosa.feature.spectral_bandwidth(y=waveform, sr=16000, n_fft=512, hop_length=hop),
             librosa.feature.spectral_rolloff(y=waveform, sr=16000, n_fft=512, hop_length=hop),
             librosa.feature.spectral_flatness(y=waveform, n_fft=512, hop_length=hop)]
    values = np.concatenate([mfcc, delta, *specs], 0).T.astype(np.float32)
    return values, np.arange(len(values)) * hop / 16000


def visual_features(video, sample_fps=5):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise ValueError(f'Cannot open video {video.name}')
    fps, count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0: raise ValueError('Invalid video FPS')
    # OpenCV's filename loader on Windows does not handle every Unicode path.
    cascade_xml = Path(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml').read_text(encoding='utf-8')
    storage = cv2.FileStorage(cascade_xml, cv2.FILE_STORAGE_READ | cv2.FILE_STORAGE_MEMORY)
    detector = cv2.CascadeClassifier()
    detector.read(storage.getFirstTopLevelNode())
    storage.release()
    if detector.empty(): raise ValueError('Face detector could not be loaded')
    previous, features, times, face_flags = None, [], [], []
    step = max(1, int(round(fps / sample_fps)))
    for index in range(0, count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok: continue
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = min(1., 480 / w)
        small = cv2.resize(gray, None, fx=scale, fy=scale)
        boxes = detector.detectMultiScale(small, 1.1, 4, minSize=(24, 24))
        if len(boxes):
            box = max(boxes, key=lambda b: b[2] * b[3]) / scale
            x, y, bw, bh = np.asarray(box, dtype=int)
            roi = frame[y:y + bh, x:x + bw]
            face_flags.append(True)
        else:
            x, y, bw, bh, roi = 0, 0, w, h, frame
            face_flags.append(False)
        face = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (32, 32)).astype(np.float32) / 255
        dct = cv2.dct(face)[:4, :4].flatten()
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32) / np.array([180, 255, 255])
        flow_stats = np.zeros(7)
        if previous is not None:
            flow = cv2.calcOpticalFlowFarneback(previous * 255, face * 255, None, .5, 2, 9, 2, 5, 1.1, 0)
            mag = np.linalg.norm(flow, axis=-1)
            flow_stats = np.r_[flow.mean((0, 1)), flow.std((0, 1)), mag.mean(), mag.std(), np.quantile(mag, .9)]
        values = np.r_[dct, hsv.mean(0), hsv.std(0), [x / w, y / h, bw / w, bh / h],
                       face.mean(), face.std(), flow_stats]
        features.append(values)
        times.append(index / fps)
        previous = face
    cap.release()
    if not features: raise ValueError('No decodable frames')
    return np.asarray(features, dtype=np.float32), np.asarray(times), face_flags, fps


def pool_spans(values, times, spans):
    out = np.zeros((len(spans), values.shape[1]), dtype=np.float32)
    for i, (left, right) in enumerate(spans):
        if right <= left or not len(values): continue
        selected = (times >= left) & (times < right)
        if not selected.any(): selected[np.argmin(np.abs(times - (left + right) / 2))] = True
        out[i] = values[selected].mean(0)
    return out


def q1(args):
    torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizerFast.from_pretrained(str(ROOT / 'pretrained' / 'bert'), local_files_only=True)
    text_encoder = FrozenText(ROOT / 'pretrained' / 'bert').to(device)
    aligner = EvidenceAligner(device)
    label_path = next(Path(args.data).rglob('label-100.xlsx'))
    labels = pd.read_excel(label_path)
    output = Path(args.output) / 'q1'
    output.mkdir(parents=True, exist_ok=True)
    videos = {(p.parent.name, p.stem): p for p in label_path.parent.rglob('*.mp4')}
    summary, index = [], []
    for _, row in labels.iterrows():
        vid, clip = str(row.video_id), str(int(row.clip_id))
        sid = f'{vid}$_${clip}'
        video = videos[(vid, clip)]
        feature_file = output / 'features' / f'{sid}.npz'
        metadata_file = output / 'alignment' / f'{sid}.json'
        feature_file.parent.mkdir(exist_ok=True)
        entry = {'sample_id': sid, 'label': float(row.label), 'annotation': row.annotation,
                 'text_dim': 768, 'audio_dim': 32, 'vision_dim': 35, 'max_length': 50}
        try:
            waveform = read_audio(video)
            mapping = align_tokens(row.text, waveform, tokenizer, aligner)
            bert = np.stack([mapping['input_ids'], mapping['attention_mask'], np.zeros(50, dtype=np.int64)])
            text = text_features(text_encoder, torch.tensor(bert[None], device=device), device)[0].cpu().numpy()
            af, at = audio_features(waveform)
            vf, vt, faces, fps = visual_features(video)
            audio = pool_spans(af, at, mapping['token_spans'])
            vision = pool_spans(vf, vt, mapping['token_spans'])
            valid = np.array([b > a for a, b in mapping['token_spans']])
            text[~valid], audio[~valid], vision[~valid] = 0, 0, 0
            np.savez_compressed(feature_file, text=text.astype(np.float16), audio=audio.astype(np.float16),
                                vision=vision.astype(np.float16), valid=valid,
                                token_ids=bert[0].astype(np.int32), spans=np.asarray(mapping['token_spans'], dtype=np.float32))
            mapping.update(sample_id=sid, video=f'{vid}/{clip}.mp4', original_text=row.text,
                           face_detection_rate=float(np.mean(faces)), fps=fps,
                           tokenization_truncated=len(tokenizer(row.text)['input_ids']) > 50)
            save_json(metadata_file, mapping)
            entry.update(duration=mapping['duration'], valid_length=int(valid.sum()),
                         alignment=mapping['status'], face_detection_rate=mapping['face_detection_rate'], status='ok')
        except Exception as exc:
            # Never silently delete a required sample.
            np.savez_compressed(feature_file, text=np.zeros((50, 768), np.float16),
                                audio=np.zeros((50, 32), np.float16), vision=np.zeros((50, 35), np.float16),
                                valid=np.zeros(50, bool), spans=np.zeros((50, 2), np.float32))
            entry.update(status='failed_preserved_with_mask', error=str(exc))
            save_json(metadata_file, entry)
        summary.append(entry)
        index.append({'sample_id': sid, 'features': str(feature_file.relative_to(output)),
                      'metadata': str(metadata_file.relative_to(output))})
        pd.DataFrame(summary).to_csv(output / 'sample_summary.csv', index=False, encoding='utf-8-sig')
        print(f'Q1 {len(summary)}/{len(labels)} {sid}: {entry["status"]}', flush=True)
    save_json(output / 'index.json', index)
    save_json(output / 'feature_definition.json', {
        'text': 'Frozen bert-base-uncased final layer; WordPiece positions; float16 storage',
        'audio': '13 MFCC + 13 first deltas + RMS/ZCR/centroid/bandwidth/rolloff/flatness; 32 dimensions',
        'vision': '16 face-region DCT + 6 HSV statistics + 4 normalized box coordinates + 2 grayscale statistics + 7 flow statistics; 35 dimensions',
        'face_fallback': 'Use whole frame when Haar detector fails; detection rate is reported',
        'alignment': 'CTC transcript forced alignment, WordPiece offset mapping, interval mean pooling',
        'scope': 'Q1 descriptive emotion-related proxies; NOT interchangeable with attachment2 acoustic/visual dimensions',
        'versions': {'torch': torch.__version__, 'opencv': cv2.__version__, 'librosa': librosa.__version__}})


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', default=str(DATA))
    p.add_argument('--output', default=str(ROOT / 'outputs'))
    q1(p.parse_args())
