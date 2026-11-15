import pytest
from aniflive_tts.expression import ConditioningPolicy, PreparedReference, PreparedReferenceBank
from aniflive_tts.backend.v5.conditioning import resolve_conditioning


@pytest.mark.parametrize('policy,roles', [
    ('full-switch', ('style', 'style', 'style')),
    ('identity-lock', ('style', 'style', 'base')),
    ('semantic-style', ('style', 'base', 'base')),
    ('acoustic-style', ('base', 'style', 'base')),
    ('sv-only', ('base', 'base', 'style')),
])
def test_native_components_follow_shared_reference_selection(policy, roles):
    bank=PreparedReferenceBank(identity_id='base')
    for name in ('base','style'):
        bank.add(PreparedReference(object(),object(),object(),[],
            {'prompt':{'owner':name},'reference':tuple(f'{name}:{i}' for i in range(7)),
             'path':name,'text':name,'language':'ja'},id=name))
    selected=resolve_conditioning(bank.conditioning(reference_id='style',
                                  policy=ConditioningPolicy(policy)),bank)
    semantic,style,voice=roles
    assert selected.bert['prompt'] is bank.get(semantic).bert['prompt']
    reference=selected.bert['reference']
    assert reference[:3] == bank.get(semantic).bert['reference'][:3]
    assert reference[3] == f'{style}:3'
    assert reference[5] == f'{style}:5'
    assert reference[4] == f'{voice}:4'
    assert reference[6] == f'{voice}:6'
    if policy=='full-switch':
        assert selected is bank.get('style')
