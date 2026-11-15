import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from aniflive_tts import api, cli, webui


@pytest.mark.parametrize('backend', ['hybrid-cuda', 'MLX-Metal'])
def test_webui_proxies_v5_pcm_with_actual_backend_and_sample_rate(backend, tmp_path, monkeypatch):
    monkeypatch.setenv('ANIFLIVE_TTS_WORKSTATION_DIR', str(tmp_path/'state'))
    monkeypatch.setenv('ANIFLIVE_TTS_WEBUI_TRUSTED_HOSTS', 'testserver')
    (tmp_path/'index.html').write_text('<!doctype html>')
    config = dict(model='v5-test', version='v5turbo', backend=backend, sample_rate=48000,
                  engine_count=5, voice='packaged' if backend=='MLX-Metal' else '',
                  reference_configured=True, pytorch_fallback=False)
    def handler(request):
        path=request.url.path
        values={
            '/openapi.json': {'info': {'version': '1.5.0'}},
            '/health': {'ready': True, 'model': 'v5-test', 'backend': backend},
            '/model/config': config,
            '/v1/models': {'data': [{'id': 'v5-test', 'active': True}]},
            '/v1/expressions': {'enabled': False, 'profiles': [], 'policies': ['full-switch']},
            '/v1/voices': {'data': [{'id': 'registered', 'default_reference_id': 'natural'},
                                  {'id': 'packaged', 'reference_configured': True}]},
            '/v1/voices/registered/references/natural': {'prepare_state': 'ready'},
        }
        if path in values:
            return httpx.Response(200, json=values[path])
        if path=='/v1/audio/cancel':
            return httpx.Response(200,json={'cancelled':False})
        if path=='/v1/audio/speech':
            assert json.loads(request.content)['voice_profile']=='registered'
            return httpx.Response(200,content=b'\x01\x00'*480,headers={
                'x-tts-backend':backend,'x-pytorch-fallback':'false','x-tts-model':'v5-test',
                'x-tts-stream':'pcm_s16le','x-tts-sample-rate':'48000','x-tts-channels':'1',
                'x-tts-recommended-prebuffer-ms':'32'})
        raise AssertionError(path)
    upstream=httpx.AsyncClient(base_url='http://upstream',transport=httpx.MockTransport(handler))
    app=webui.create_webui_app(static_dir=tmp_path,client=upstream)
    with TestClient(app) as client:
        status=client.get('/api/status')
        assert status.status_code==200 and status.json()['ready']
        assert len(status.json()['voices'])==2
        response=client.post('/api/speech',json={
            'text':'今日はいい天気ですね。','language':'ja','model':'v5-test',
            'voice_profile':'registered'})
        assert response.status_code==200, response.content
        assert response.headers['x-tts-sample-rate']=='48000'
        assert response.headers['x-tts-backend']==backend
        assert response.content==b'\x01\x00'*480


def test_cli_native_serve_preserves_shared_official_frontend(tmp_path,monkeypatch):
    package=tmp_path/'package'
    package.mkdir()
    (package/'manifest.json').write_text(json.dumps({'native_runtime':{}}))
    monkeypatch.delenv('ANIFLIVE_TTS_SOURCE_DIR',raising=False)
    monkeypatch.setattr(api,'create_app',lambda:object())
    import uvicorn
    monkeypatch.setattr(uvicorn,'run',lambda *args,**kwargs:None)
    cli._serve(SimpleNamespace(model_package=package,shared_dir=tmp_path/'shared',host='127.0.0.1',port=9880))
    import os
    assert os.environ['ANIFLIVE_TTS_SOURCE_DIR']==str((tmp_path/'shared'/'official').resolve())
