"""
wav2vec ASR Module
==================

This module provides on-device ASR capabilities by using the wav2vec2 transformer
provided by huggingface. In addition, the ASR module provides end-of-utterance detection
(with a VAD), so that produced hypotheses are being "committed" once an utterance is
finished.
"""

import threading
from retico_core import *
from retico_core.audio import AudioIU
from retico_core.text import SpeechRecognitionIU
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import transformers
import pydub
import webrtcvad
import numpy as np
import time

transformers.logging.set_verbosity_error()


class Wav2Vec2ASR:
    def __init__(
        self,
        wav2vec2_model="facebook/wav2vec2-base-960h",
        framerate=16_000,
        silence_dur=1,
        vad_agressiveness=3,
        silence_threshold=0.75,
    ):
        self.processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model)
        self.model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model)
        self.model.freeze_feature_encoder()
        self.audio_buffer = []
        self.framerate = framerate
        self.vad = webrtcvad.Vad(vad_agressiveness)
        self.silence_dur = silence_dur
        self.vad_state = False
        self._n_sil_frames = None
        self.silence_threshold = silence_threshold

    def _resample_audio(self, audio):
        if self.framerate != 16_000:
            # If the framerate is not 16 kHz, we need to resample
            s = pydub.AudioSegment(
                audio, sample_width=2, channels=1, frame_rate=self.framerate
            )
            s = s.set_frame_rate(16_000)
            return s._data
        return audio

    def get_n_sil_frames(self):
        if not self._n_sil_frames:
            if len(self.audio_buffer) == 0:
                return None
            frame_length = len(self.audio_buffer[0]) / 2
            self._n_sil_frames = int(self.silence_dur / (frame_length / 16_000))
        return self._n_sil_frames

    def recognize_silence(self):
        n_sil_frames = self.get_n_sil_frames()
        if not n_sil_frames or len(self.audio_buffer) < n_sil_frames:
            return True
        silence_counter = 0
        for a in self.audio_buffer[-n_sil_frames:]:
            if not self.vad.is_speech(a, 16_000):
                silence_counter += 1
        if silence_counter >= int(self.silence_threshold * n_sil_frames):
            return True
        return False

    def add_audio(self, audio):
        audio = self._resample_audio(audio)
        self.audio_buffer.append(audio)

    def recognize(self):
        silence = self.recognize_silence()

        if not self.vad_state and not silence:
            self.vad_state = True
            self.audio_buffer = self.audio_buffer[-self.get_n_sil_frames() :]

        if not self.vad_state:
            return None, False

        full_audio = b""
        for a in self.audio_buffer:
            full_audio += a
        npa = np.frombuffer(full_audio, dtype=np.int16).astype(np.double)
        if len(npa) < 10:
            return None, False
        input_values = self.processor(
            npa, return_tensors="pt", sampling_rate=16000
        ).input_values
        logits = self.model(input_values).logits
        predicted_ids = np.argmax(logits.detach().numpy(), axis=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0].lower()

        if silence:
            self.vad_state = False
            self.audio_buffer = []

        return transcription, self.vad_state

    def reset(self):
        self.vad_state = True
        self.audio_buffer = []


class Wav2VecASRModule(AbstractModule):
    @staticmethod
    def name():
        return "Wav2Vec ASR Module"

    @staticmethod
    def description():
        return "A module that recognizes speech using Wav2Vec."

    @staticmethod
    def input_ius():
        return [AudioIU]

    @staticmethod
    def output_iu():
        return SpeechRecognitionIU

    LANGUAGE_MAPPING = {
        "en": "facebook/wav2vec2-base-960h",
        "de": "oliverguhr/wav2vec2-large-xlsr-53-german-cv9",
        "fr": "facebook/wav2vec2-large-xlsr-53-french",
        "es": "facebook/wav2vec2-large-xlsr-53-spanish",
    }

    def __init__(self, language="en", framerate=None, silence_dur=1, **kwargs):
        super().__init__(**kwargs)

        if language not in self.LANGUAGE_MAPPING.keys():
            print("Unknown ASR language. Defaulting to English (en).")
            language = "en"

        self.language = language
        self.acr = Wav2Vec2ASR(
            wav2vec2_model=self.LANGUAGE_MAPPING[language],
            silence_dur=silence_dur,
        )
        self.framerate = framerate
        self.silence_dur = silence_dur
        self._asr_thread_active = False
        self.latest_input_iu = None

    def process_update(self, update_message):
        for iu, ut in update_message:
            # Audio IUs are only added and never updated.
            if ut != UpdateType.ADD:
                continue
            if self.framerate is None:
                self.framerate = iu.rate
                self.acr.framerate = self.framerate
            self.acr.add_audio(iu.raw_audio)
            if not self.latest_input_iu:
                self.latest_input_iu = iu

    def _asr_thread(self):
        while self._asr_thread_active:
            time.sleep(0.5)
            prediction, vad = self.acr.recognize()
            end_of_utterance = not vad and prediction is not None
            if prediction is None:
                continue
            um, new_tokens = self.get_increment(prediction)

            if len(new_tokens) == 0:
                if vad:
                    continue
                else:
                    output_iu = self.create_iu(self.latest_input_iu)
                    output_iu.set_asr_results([prediction], "", 1.0, 0.99, True)
                    output_iu.committed = True
                    self.current_ius = []
                    um.add_iu(output_iu, UpdateType.ADD)

            for i, token in enumerate(new_tokens):
                output_iu = self.create_iu(self.latest_input_iu)
                eou = i == len(new_tokens) - 1 and end_of_utterance
                output_iu.set_asr_results([prediction], token, 0.0, 0.99, eou)
                if eou:
                    output_iu.committed = True
                    self.current_ius = []
                else:
                    self.current_ius.append(output_iu)
                um.add_iu(output_iu, UpdateType.ADD)

            self.latest_input_iu = None
            self.append(um)

    def get_increment(self, new_text):
        """Compares the full text given by the asr with the IUs that are already
        produced and returns only the increment from the last update. It revokes all
        previously produced IUs that do not match."""
        um = UpdateMessage()
        tokens = new_text.strip().split(" ")
        if tokens == [""]:
            return um, []

        new_tokens = []
        iu_idx = 0
        token_idx = 0
        while token_idx < len(tokens):
            if iu_idx >= len(self.current_ius):
                new_tokens.append(tokens[token_idx])
                token_idx += 1
            else:
                current_iu = self.current_ius[iu_idx]
                iu_idx += 1
                if tokens[token_idx] == current_iu.text:
                    token_idx += 1
                else:
                    current_iu.revoked = True
                    um.add_iu(current_iu, UpdateType.REVOKE)
        self.current_ius = [iu for iu in self.current_ius if not iu.revoked]

        return um, new_tokens

    def prepare_run(self):
        self._asr_thread_active = True
        threading.Thread(target=self._asr_thread).start()

    def shutdown(self):
        self._asr_thread_active = False
        self.acr.reset()
