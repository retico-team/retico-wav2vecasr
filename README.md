# retico-wav2vecasr

A ReTico module for automatic speech recognition (ASR) using Meta's Wav2Vec Model. The module processes audio from microphone, converts speech into text using the model's pretrained transformer, and finally outputs transcriptions using ReTico's framework. 

## Installation

To use the Automatic Speech Recognition module based on Wav2Vec, you first need to install the retico-core package:

* Install the package ```pip install git+ git+https://github.com/retico-team/retico-core```

Right after that, install the wav2vecasr package:

* Install the package ```pip install git+ git+https://github.com/retico-team/retico-wav2vecasr```

Dependencies including PyTorch will be installed automatically. However, depending on your system (CPU/GPU), you may need to install the correct PyTorch build manually from https://pytorch.org.

## Module

### `Wav2VecASRModule`
This module performs ASR using a pretrained Wav2Vec2 model from Hugging Face. It processes the audio incrementally and produces audio transcriptions. The module also uses internal setting for speech detections and create end-of-speech.

**Model options:** `en`(English), `de`(German), `fr`(French), `es`(Spanish)

#### Arguments:
* `language` (str): Language to be used, current languages are limited to 'en', 'de', 'fr', 'es'
* `framerate` (int): Sample rate, defaults to audio IU
* `silence_dur` (float): Time before it identifies end-of-speech, defaults to 1 second

## Example

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


## Citation

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
