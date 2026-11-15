"""CPU-only content regression gate; uses product CER/WER normalization."""
import argparse
import math
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from faster_whisper import WhisperModel
import numpy as np
import soundfile as sf
import soxr


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--text',required=True)
    p.add_argument('--language',choices=['ja','en','zh','yue','ko'],required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--whisper-language',choices=['ja','en','zh','yue','ko'])
    p.add_argument('--max-content-error',type=float,default=0.15)
    a=p.parse_args()
    if not math.isfinite(a.max_content_error) or not 0<=a.max_content_error<=1:
        raise ValueError('max-content-error must be finite between 0 and 1')
    sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
    from aniflive_tts.workstation_evaluation import content_error, orthographic_content_error
    lock_path=a.model/'source-lock.json'
    lock=json.loads(lock_path.read_text())
    fingerprints={name:sha(a.model/name) for name in lock['files']}
    for name,actual in fingerprints.items():
        if actual!=lock['files'][name]['sha256']:
            raise ValueError(f'ASR model checksum mismatch {name}')
    # CPU f32 evaluation is explicit and separate from the native TTS engine.
    model=WhisperModel(str(a.model.resolve()),device='cpu',compute_type='float32',cpu_threads=4,num_workers=1,local_files_only=True)
    options={'language':a.whisper_language or ('zh' if a.language=='yue' else a.language),'task':'transcribe','beam_size':5,'temperature':0.0,'vad_filter':False,'condition_on_previous_text':False}
    rows={}
    for key,path in [('reference',a.reference),('candidate',a.candidate)]:
        audio,rate=sf.read(path,dtype='float32',always_2d=True)
        mono=audio.mean(axis=1,dtype=np.float32)
        if rate!=16000:
            mono=soxr.resample(mono,rate,16000,quality='HQ')
        if not mono.size or not np.isfinite(mono).all():
            raise ValueError('Invalid ASR input audio')
        segments,info=model.transcribe(mono,**options)
        text=''.join(segment.text for segment in segments).strip()
        rows[key]={'path':str(path.resolve()),'sha256':sha(path),'transcript':text,'raw_content':content_error(a.text,text,a.language),'content':orthographic_content_error(a.text,text,a.language),'language_probability':float(info.language_probability)}
    # Baseline parity alone cannot qualify unrelated speech; require both
    # no regression and independently bounded absolute content error.
    delta=rows['candidate']['content']['error_rate']-rows['reference']['content']['error_rate']
    regression_passed=bool(rows['reference']['transcript'] and rows['candidate']['transcript'] and delta<=0)
    absolute_passed=all(row['content']['error_rate']<=a.max_content_error for row in rows.values())
    passed=regression_passed and absolute_passed
    for name,actual in fingerprints.items():
        if sha(a.model/name)!=actual:
            raise ValueError(f'ASR model changed during evaluation {name}')
    report={'scope':'one controlled utterance ASR regression only; not full quality qualification','expected_text':a.text,'language':a.language,'model':{'repository':lock['repository'],'revision':lock['revision'],'source_lock_sha256':sha(lock_path),'files_sha256':fingerprints},'backend':{'name':'faster-whisper','version':importlib.metadata.version('faster-whisper'),'ctranslate2_version':importlib.metadata.version('ctranslate2'),'device':'cpu','compute_type':'float32'},'options':options,'audio_preprocessing':'soundfile float32 mono mean; soxr HQ resampling to 16 kHz; supported ndarray ASR input for every file','results':rows,'error_rate_delta':delta,'gate':{'max_error_rate_delta':0.0,'nonempty_required':True,'regression_passed':regression_passed,'absolute_content_error_max':a.max_content_error,'absolute_content_passed':absolute_passed,'passed':passed},'limitations':['ASR is an objective proxy, not human listening acceptance','One model/utterance; cross-language and multi-seed regressions remain required',('Cantonese decoder uses explicitly selected yue; broader dialect regression remains required' if a.language=='yue' and options['language']=='yue' else 'Cantonese zh decoding, when selected, needs separate dialect-aware verification')]}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'results':rows,'error_rate_delta':delta,'passed':passed},ensure_ascii=False))
    return 0 if passed else 1

if __name__=='__main__':
    raise SystemExit(main())
