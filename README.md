# retico-wav2vecasr
Local wav2vec see citation below for modle information) ASR Module for ReTiCo.

### Installations and requirements

you can install the module via pip:

```bash
$ pip install retico-wav2vecasr
```

In order to access the ASR models, one of PyTorch, TensorFlow, or Flax need to
be installed. For example, PyTorch can be installed via pip with:

```bash
$ pip install torch
```

### Example

```python
import retico_core
from retico_wav2vecasr import *
from retico_wav2vecasr.wav2vecasr import Wav2VecASRModule

msg = []


def callback(update_msg):
    global msg
    for x, ut in update_msg:
        if ut == retico_core.UpdateType.ADD:
            msg.append(x)
        if ut == retico_core.UpdateType.REVOKE:
            msg.remove(x)
    txt = ""
    committed = False
    for x in msg:
        txt += x.text + " "
        committed = committed or x.committed
    print(" " * 80, end="\r")
    print(f"{txt}", end="\r")
    if committed:
        msg = []
        print("")


microphone = retico_core.audio.MicrophoneModule()
asr = Wav2VecASRModule("en")


m3 = debug.CallbackModule(callback=callback)

microphone.subscribe(asr)
asr.subscribe(m3)

retico_core.network.run(asr)

print("Running the ASR. Press enter to exit")
input()

retico_core.network.stop(asr)
```


Citation

```
@misc{https://doi.org/10.48550/arxiv.2006.11477,
  doi = {10.48550/ARXIV.2006.11477},
  
  url = {https://arxiv.org/abs/2006.11477},
  
  author = {Baevski, Alexei and Zhou, Henry and Mohamed, Abdelrahman and Auli, Michael},
  
  keywords = {Computation and Language (cs.CL), Machine Learning (cs.LG), Sound (cs.SD), Audio and Speech Processing (eess.AS), FOS: Computer and information sciences, FOS: Computer and information sciences, FOS: Electrical engineering, electronic engineering, information engineering, FOS: Electrical engineering, electronic engineering, information engineering},
  
  title = {wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations},
  
  publisher = {arXiv},
  
  year = {2020},
  
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
