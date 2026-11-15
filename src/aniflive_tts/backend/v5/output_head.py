"""Opt-in existing AnifLive-TTS v2 head-silence output contract, without sample resynthesis."""
import numpy as np
from aniflive_tts.streaming import TensorRTFixedReferenceStreamer as Existing
from aniflive_tts.streaming import INITIAL_LEADING_SILENCE_RETAIN_SECONDS


class V2Head:
    def __init__(self,rate=48000,retained_seconds=INITIAL_LEADING_SILENCE_RETAIN_SECONDS):
        self.rate=rate
        self.done=False
        self.discarded_samples=0
        self.retained_seconds=retained_seconds

    def feed(self,pcm):
        if self.done or not pcm.size:
            return pcm
        audio=pcm.astype(np.float32)/32768.0
        leading=Existing._leading_silence_seconds(audio,self.rate)
        if self.rate <= 0 or not leading < float(audio.size)/float(self.rate):
            self.discarded_samples+=pcm.size
            return pcm[:0]
        processed,_=Existing._trim_excess_leading_silence(audio,sample_rate=self.rate,
            leading_silence_seconds=leading,retained_silence_seconds=self.retained_seconds)
        count=pcm.size-processed.size
        self.discarded_samples+=count
        self.done=True
        return pcm[count:]
