# retico-wav2vecasr
Local wav2vec ASR Module for ReTiCo

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
