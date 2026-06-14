import numpy as np
import scipy.io
from pathlib import Path
import sys

_scr = Path(__file__).resolve().parents[1]
if str(_scr) not in sys.path:
    sys.path.insert(0, str(_scr))

from pipeline.pipeline import SSVEPPipeline

mat = scipy.io.loadmat("data/raw/S1.mat")
data = mat["data"]  

data = data.transpose(2, 3, 0, 1)  
data = data.reshape(-1, 64, 1500)   
data = data[:, :, 125:1375]         

freq_phase = scipy.io.loadmat("data/raw/Freq_Phase.mat")
frequencies_hz = freq_phase["freqs"].flatten().tolist()
labels = [str(round(f, 1)) for f in frequencies_hz]
true_labels = [labels[i // 6] for i in range(240)]

pipeline = SSVEPPipeline(
    frequencies_hz=frequencies_hz,
    sample_rate_hz=250.0,
    n_harmonics=7,
    notch_hz=50.0,
    bandpass_low_hz=6.0,
    bandpass_high_hz=90.0,
    labels=labels,
)
results = pipeline.evaluate(data, true_labels)

print(f"Accuracy: {results['accuracy']:.1%} ({results['n_correct']}/{results['n_total']})")
