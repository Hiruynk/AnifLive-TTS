from __future__ import annotations
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
from pathlib import Path
import sys
import threading
from types import SimpleNamespace
import wave
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_release import WavPCM, request
from benchmark_audio import analyze_pcm16_stream


def audio(rate):
    pcm=(np.sin(np.arange(rate//10)*.1)*10000).astype('<i2').tobytes()
    f=io.BytesIO()
    with wave.open(f,'wb') as w:
        w.setparams((1,2,rate,0,'NONE','not compressed'))
        w.writeframes(pcm)
    return f.getvalue(),pcm


@pytest.mark.parametrize('rate',[32000,48000])
def test_streaming_wav_header_does_not_count_as_pcm(rate):
    wav,pcm=audio(rate)
    decoder=WavPCM(rate)
    assert decoder.feed(wav[:7])==b''
    assert decoder.feed(wav[7:44])==b''
    actual=decoder.feed(wav[44:])
    assert actual==pcm
    result=analyze_pcm16_stream([(.2,actual)],rate)
    assert result.ttfp_seconds==.2
    assert result.audible_ttfa_seconds>=.2


def test_mismatched_model_rate_is_rejected():
    wav,_=audio(48000)
    with pytest.raises(ValueError,match='mismatched'):
        WavPCM(32000).feed(wav)


def test_real_http_client_active_close_and_pool_lifecycle():
    _,pcm=audio(32000)
    class Handler(BaseHTTPRequestHandler):
        protocol_version='HTTP/1.1'
        def log_message(self,*args): pass
        def do_POST(self):
            self.rfile.read(int(self.headers.get('Content-Length',0)))
            assert self.headers['Connection']=='keep-alive'
            self.send_response(200)
            self.send_header('Content-Length',str(len(pcm)))
            self.send_header('X-TTS-Sample-Rate','32000')
            self.end_headers()
            self.wfile.write(pcm)
    server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    a=SimpleNamespace(host='127.0.0.1',port=server.server_port,timeout=3,path='/tts',protocol='pcm',sample_rate=32000)
    pool=http.client.HTTPConnection(a.host,a.port,timeout=3)
    try:
        for _ in range(2):
            row,_,actual=request(a,b'{}','new_connection_pcm')
            assert not row['socket_reused_at_start'] and row['socket_open_after_response']
            assert actual==pcm
        first,_,_=request(a,b'{}','keepalive_pcm',pool)
        second,_,_=request(a,b'{}','keepalive_pcm',pool)
        assert not first['socket_reused_at_start'] and second['socket_reused_at_start']
        assert first['local_address']==second['local_address']
    finally:
        pool.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
