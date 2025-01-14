# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/C. Word error rate metrics.ipynb.

# %% auto 0
__all__ = ['librispeech_data', 'DfBuilder', 'WERStats']

# %% ../nbs/C. Word error rate metrics.ipynb 2
import jiwer
from whisper_normalizer.english import EnglishTextNormalizer

import torchaudio
from pathlib import Path
import pandas as pd

# %% ../nbs/C. Word error rate metrics.ipynb 3
engnorm = EnglishTextNormalizer()
def whisper_normalize(x):
    if type(x) == list:
        return [engnorm(y) for y in x]
    else:
        return engnorm(x)

default_transform = jiwer.transforms.Compose([
    jiwer.transforms.ToLowerCase(),
    jiwer.transforms.RemoveMultipleSpaces(),
    jiwer.transforms.Strip(),
    jiwer.transforms.RemovePunctuation(),
    jiwer.transforms.ReduceToListOfListOfWords(),
])

# %% ../nbs/C. Word error rate metrics.ipynb 5
def librispeech_data(datadir, sample_rate=16000):
    for file in Path(datadir).rglob('*.txt'):
        for line in file.read_text().split('\n'):
            if not line: continue
            idx, text = line.split(" ", 1)
            x, sr = torchaudio.load((file.parent/idx).with_suffix('.flac'))
            if sr != sample_rate:
                x = torchaudio.transforms.Resample(sr, self.sample_rate)(x)
            yield x, text

# %% ../nbs/C. Word error rate metrics.ipynb 6
class DfBuilder:
    def __init__(self):
        self.data = {}
        
    def push(self, **kwargs):
        for k,v in kwargs.items():
            if k not in self.data:
                self.data[k] = [v]
            else:
                self.data[k].append(v)
    
    def df(self):
        return pd.DataFrame(self.data)

# %% ../nbs/C. Word error rate metrics.ipynb 7
class WERStats(DfBuilder):
    def __init__(self, transform=default_transform):
        super().__init__()
        self.reference_transform = transform
        self.hypothesis_transform = transform
    
    def push_sample(self, snd, gt_text, text, idx=None):
        if snd is not None: self.push(secs = snd.shape[-1]/16000)
        diff = jiwer.process_words(gt_text, text, reference_transform=self.reference_transform, hypothesis_transform=self.hypothesis_transform)
        self.push(
            idx = idx,
            gt_text = gt_text,
            text = text,
            wer = diff.wer,
            mer = diff.mer,
            wil = diff.wil,
            wip = diff.wip,
        )
        return diff