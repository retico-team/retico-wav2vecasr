# retico-wav2vecasr
Local wav2vec ASR Module for ReTiCo


### Example

```python
import retico_core
from retico_wav2vecasr import *
from retico_wav2vecasr.wav2vecasr import Wav2VecASRModule

microphone = retico_core.audio.MicrophoneModule(960, 48000)
asr = Wav2VecASRModule("de")

msg = []


def callback(update_msg):
    global msg
    for x, ut in update_msg:
        if ut == retico_core.UpdateType.ADD:
            msg.append(x)
        if ut == retico_core.UpdateType.REVOKE:
            if x not in msg:
                print("ERROR", x, msg)
            msg.remove(x)
    txt = ""
    committed = False
    for x in msg:
        txt += x.text
        committed = committed or x.committed
    print(f"                                                ", end="\r")
    print(f"{txt}", end="\r")
    if committed:
        msg = []
        print("")


m3 = debug.CallbackModule(callback=callback)

microphone.subscribe(asr)
asr.subscribe(m3)

retico_core.network.run(asr)

print("Running the stuff")
input()

retico_core.network.stop(asr)
```