#!/usr/bin/env python3
"""Serial release HTTP benchmark for raw PCM and chunked PCM16 WAV services.

Bodies and references are supplied by the operator, never bundled. Each fresh
request is actively closed by its client after its complete response is read.
"""
from __future__ import annotations
import argparse
import hashlib
import http.client
import io
import json
from pathlib import Path
import statistics
import struct
import subprocess
import threading
import time
import wave
from benchmark_audio import analyze_pcm16_stream, AUDIBLE_FRAME_MS, AUDIBLE_THRESHOLD_DBFS
from benchmark_readme import percentile


def digest(data): return hashlib.sha256(data).hexdigest()


class WavPCM:
    """Incrementally remove RIFF metadata; retain real PCM arrival times."""
    def __init__(self, rate):
        self.rate, self.buffer, self.format, self.data = rate, bytearray(), None, False
        self.riff = False

    def feed(self, payload):
        if self.data:
            return payload
        self.buffer.extend(payload)
        if not self.riff:
            if len(self.buffer) < 12:
                return b''
            if self.buffer[:4] != b'RIFF' or self.buffer[8:12] != b'WAVE':
                raise ValueError('Expected RIFF/WAVE stream')
            del self.buffer[:12]
            self.riff = True
        while len(self.buffer) >= 8:
            kind, count = struct.unpack_from('<4sI', self.buffer)
            if kind == b'data':
                if self.format != (1, 1, self.rate, 16):
                    raise ValueError(f'Unsupported or mismatched WAV format: {self.format}')
                self.data = True
                result = bytes(self.buffer[8:])
                self.buffer.clear()
                return result
            end = 8 + count + (count & 1)
            if len(self.buffer) < end:
                return b''
            if kind == b'fmt ':
                if count < 16:
                    raise ValueError('Short WAV format chunk')
                fmt, channels, rate, _, _, bits = struct.unpack_from('<HHIIHH', self.buffer, 8)
                self.format = (fmt, channels, rate, bits)
            del self.buffer[:end]
        return b''


class GPUUsage:
    def __init__(self, enabled, output):
        self.enabled, self.output = enabled, output
        self.rows, self.error, self.done = [], None, threading.Event()

    def start(self):
        if self.enabled:
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def run(self):
        fields = ['utilization.gpu', 'memory.used', 'clocks.sm', 'temperature.gpu', 'power.draw']
        try:
            with self.output.open('x', encoding='utf-8') as out:
                while not self.done.is_set():
                    value = subprocess.check_output(['nvidia-smi', '--query-gpu='+','.join(fields),
                        '--format=csv,noheader,nounits'], text=True, timeout=5).splitlines()[0]
                    values = [x.strip() for x in value.split(',')]
                    row = dict(utc=time.time())
                    for name, value in zip(fields, values):
                        row[name] = None if value in ('N/A', '[N/A]', '[Not Supported]') else float(value)
                    self.rows.append(row)
                    out.write(json.dumps(row)+'\n')
                    out.flush()
                    self.done.wait(1)
        except Exception as error:
            self.error = repr(error)

    def close(self):
        if self.enabled:
            self.done.set()
            self.thread.join(timeout=6)
            if self.error or not self.rows:
                raise RuntimeError('GPU telemetry failed: '+str(self.error))


def request(a, body, mode, connection=None):
    owns = connection is None
    conn = connection or http.client.HTTPConnection(a.host, a.port, timeout=a.timeout)
    row = dict(mode=mode, request_body_sha256=digest(body), socket_reused_at_start=conn.sock is not None)
    started = time.perf_counter()
    try:
        begin = time.perf_counter()
        if conn.sock is None:
            conn.connect()
        row.update(connect_seconds=time.perf_counter()-begin,
                   local_address=conn.sock.getsockname(), peer_address=conn.sock.getpeername())
        conn.request('POST', a.path, body, {'Content-Type':'application/json; charset=utf-8',
                                         'Connection':'keep-alive'})
        begin = time.perf_counter()
        response = conn.getresponse()
        row['headers_wait_seconds'] = time.perf_counter()-begin
        if response.status != 200:
            payload = response.read()
            raise RuntimeError(f'HTTP {response.status}: {payload[:300]!r}')
        raw, pcm_chunks, arrivals = [], [], []
        decoder = WavPCM(a.sample_rate) if a.protocol == 'wav' and mode != 'complete_wav' else None
        while True:
            chunk = response.read1(65536)
            if not chunk:
                break
            elapsed = time.perf_counter()-started
            raw.append(chunk)
            arrivals.append(dict(seconds=elapsed, bytes=len(chunk)))
            if mode != 'complete_wav':
                pcm = decoder.feed(chunk) if decoder else chunk
                if pcm:
                    pcm_chunks.append((elapsed, pcm))
        payload = b''.join(raw)
        wall = time.perf_counter()-started
        if mode == 'complete_wav':
            with wave.open(io.BytesIO(payload),'rb') as wav:
                if (wav.getnchannels(),wav.getsampwidth(),wav.getframerate()) != (1,2,a.sample_rate):
                    raise ValueError('Complete WAV must match mono PCM16 model sample rate')
                pcm = wav.readframes(wav.getnframes())
            audibility = None
        else:
            if decoder and not decoder.data:
                raise ValueError('Streaming WAV ended before PCM data')
            if a.protocol == 'pcm' and response.getheader('x-tts-sample-rate') != str(a.sample_rate):
                raise ValueError('PCM sample rate header mismatch')
            pcm = b''.join(data for _,data in pcm_chunks)
            if not pcm or len(pcm)%2:
                raise ValueError('Incomplete PCM16 response')
            audibility = analyze_pcm16_stream(pcm_chunks,a.sample_rate)
        row.update(wall_seconds=wall, sample_rate=a.sample_rate, samples=len(pcm)//2,
            duration_seconds=len(pcm)/2/a.sample_rate, rtf=wall/(len(pcm)/2/a.sample_rate),
            response_sha256=digest(payload), pcm_sha256=digest(pcm),
            first_body_packet_seconds=arrivals[0]['seconds'] if arrivals else None,
            first_pcm_packet_seconds=audibility.ttfp_seconds if audibility else None,
            audible_ttfa_seconds=audibility.audible_ttfa_seconds if audibility else None,
            network_arrivals=arrivals,
            pcm_arrivals=[dict(seconds=t,bytes=len(data)) for t,data in pcm_chunks],
            server_closed_after_response=response.will_close,
            socket_open_after_response=conn.sock is not None, outcome='complete')
        return row, payload, pcm
    finally:
        if owns:
            conn.close()


def summarize(rows):
    result = dict(count=len(rows))
    for key in ('wall_seconds','rtf','first_body_packet_seconds','first_pcm_packet_seconds','audible_ttfa_seconds'):
        values = [x[key] for x in rows if x.get(key) is not None]
        if values:
            result[key] = dict(p50=statistics.median(values),p95=percentile(values,.95),
                               maximum=max(values),minimum=min(values))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port',type=int,required=True)
    p.add_argument('--path', default='/v1/audio/speech')
    p.add_argument('--protocol', choices=('pcm','wav'), default='pcm')
    p.add_argument('--sample-rate',type=int,choices=(32000,48000),required=True)
    p.add_argument('--body-file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--label',required=True)
    p.add_argument('--sessions',type=int,default=10)
    p.add_argument('--warmup',type=int,default=10)
    p.add_argument('--runs',type=int,default=100)
    p.add_argument('--timeout',type=float,default=600)
    p.add_argument('--gpu-telemetry',action='store_true')
    a = p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    source = json.loads(a.body_file.read_text(encoding='utf-8-sig'))
    bodies = {}
    for stream in (False,True):
        body = {**source,'stream':stream}
        if a.protocol=='pcm':
            body['response_format']='pcm' if stream else 'wav'
        bodies[stream] = json.dumps(body,ensure_ascii=False,separators=(',',':')).encode()
    report = dict(status='running',label=a.label,protocol=a.protocol,sample_rate=a.sample_rate,
        source_sha256=digest(Path(__file__).read_bytes()),body_file_sha256=digest(a.body_file.read_bytes()),
        metric_source_sha256=digest(Path(__file__).with_name('benchmark_audio.py').read_bytes()),
        frame_ms=AUDIBLE_FRAME_MS,threshold_dbfs=AUDIBLE_THRESHOLD_DBFS,
        sessions=[],requested=dict(sessions=a.sessions,warmup=a.warmup,runs_per_mode=a.runs),
        connection_policy='Client-active-close for fresh TCP; pool reopens only before a new request if peer already closed; reconnect cost remains measured',
        samples_scope='serial real HTTP; no playback-device latency; raw body first packet and first PCM packet reported separately')
    def save():
        temp=a.output/'report.json.tmp'
        temp.write_text(json.dumps(report,indent=2),encoding='utf-8')
        temp.replace(a.output/'report.json')
    usage=GPUUsage(a.gpu_telemetry,a.output/'gpu.jsonl')
    usage.start()
    measured=[]
    counts=dict(measured=0,warmup=0,prime=0)
    try:
        with (a.output/'requests.jsonl').open('x',encoding='utf-8') as log:
            for session in range(1,a.sessions+1):
                rows=[]
                conn=http.client.HTTPConnection(a.host,a.port,timeout=a.timeout)
                try:
                    workloads=[('complete_wav',False,'warmup',a.warmup,None),
                        ('complete_wav',False,'measured',a.runs,None),
                        ('new_connection_pcm',True,'measured',a.runs,None),
                        ('keepalive_pcm',True,'prime',1,conn),
                        ('keepalive_pcm',True,'measured',a.runs,conn)]
                    for mode,stream,role,number,pool in workloads:
                        for index in range(number):
                            row,payload,pcm=request(a,bodies[stream],mode,pool)
                            row.update(session=session,role=role,index=index+1)
                            log.write(json.dumps(row)+'\n')
                            log.flush()
                            counts[role]+=1
                            if role=='measured':
                                measured.append(row)
                                rows.append(row)
                                if session==1 and index==0:
                                    (a.output/(mode+('.pcm' if stream else '.wav'))).write_bytes(pcm if stream else payload)
                            if usage.error:
                                raise RuntimeError('GPU sampler failed: '+usage.error)
                    report['sessions'].append(dict(session=session,modes={m:summarize([r for r in rows if r['mode']==m])
                        for m in ('complete_wav','new_connection_pcm','keepalive_pcm')}))
                    report['counts']=dict(counts)
                    save()
                    print(json.dumps(dict(completed_session=session,counts=counts)),flush=True)
                finally:
                    conn.close()
        usage.close()
        report.update(status='complete_formal' if (a.sessions,a.warmup,a.runs)==(10,10,100) else 'complete_exploratory',
            counts=counts,pooled={m:summarize([r for r in measured if r['mode']==m])
                for m in ('complete_wav','new_connection_pcm','keepalive_pcm')},
            gpu_samples=len(usage.rows),gpu_telemetry_enabled=a.gpu_telemetry,
            keepalive_measured_reconnects=sum(not r['socket_reused_at_start'] for r in measured if r['mode']=='keepalive_pcm'),
            rows_above_one_second=sum(r['wall_seconds']>1 for r in measured))
        save()
    except Exception as error:
        report.update(status='failed',error=repr(error),counts=counts)
        save()
        raise
    finally:
        if a.gpu_telemetry:
            usage.done.set()
            usage.thread.join(timeout=6)
    print(json.dumps(dict(status=report['status'],counts=counts,pooled=report.get('pooled'))),flush=True)


if __name__=='__main__':
    main()
