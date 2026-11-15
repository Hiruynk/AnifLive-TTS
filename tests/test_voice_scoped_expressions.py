import asyncio
from pathlib import Path
from types import SimpleNamespace
import httpx
from aniflive_tts import service,webui


def test_expression_discovery_keeps_voice_labels_and_discrete_references(monkeypatch):
    refs=[dict(id='a',emotion_id='custom_happy',emotion_label='喜悅 Amused',intensity=.3,
               transcript='Hello.',reference_language='en',transcript_confirmed=True,prepare_state='ready'),
          dict(id='b',emotion_id='custom_happy',emotion_label='喜悅 Amused',intensity=.8,
               transcript='Good morning.',reference_language='en',transcript_confirmed=True,prepare_state='unprepared')]
    store=SimpleNamespace(get_voice=lambda voice:{'default_reference_id':'a'},
        list_references=lambda voice:refs,audio_path=lambda voice,ref:Path('/reference')/ref)
    monkeypatch.setattr(service,'VOICE_LIBRARY',SimpleNamespace(store=store))
    monkeypatch.setattr(service,'VOICE_ID','')
    monkeypatch.setattr(service,'SERVICE',SimpleNamespace(expression_metadata=lambda:{
        'preferred_policy':'full-switch','policies':['full-switch']}))
    metadata=asyncio.run(service.list_expressions('voice'))
    assert metadata['voice_profile']=='voice' and metadata['enabled']
    assert metadata['profiles'][0]['intensity_levels']==[.3,.8]
    selected=webui.resolve_expression_prompt('喜悅 80%',metadata)
    assert selected.profile=='custom_happy' and selected.intensity==.8
    assert webui.resolve_expression_prompt('amused',metadata).profile=='custom_happy'


def test_webui_prepares_the_selected_emotion_reference():
    prepared=[]
    states={'a':'ready','b':'unprepared'}
    def handle(request):
        ref=request.url.path.split('/')[5]
        if request.method=='POST':
            prepared.append(ref)
            states[ref]='ready'
            return httpx.Response(202,json={})
        return httpx.Response(200,json={'prepare_state':states[ref]})
    async def run():
        async with httpx.AsyncClient(base_url='http://upstream',transport=httpx.MockTransport(handle)) as client:
            await webui._prepare_webui_voice(client,'voice',{
                'config':{'voice':''},'voices':[{'id':'voice','default_reference_id':'a'}]},
                [webui.ResolvedExpression(enabled=True,profile='happy',intensity=.9)],
                {'profiles':[{'id':'happy','references':[{'id':'a','intensity':.3},{'id':'b','intensity':.8}]}]})
    asyncio.run(run())
    assert prepared==['b']
