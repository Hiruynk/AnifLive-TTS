"""Map the shared reference-selection contract onto V5's native components.

V5 has no V2ProPlus SV encoder. Its native adapter uses GE/spectrum for style
and CFM reference features/mel for the voice anchor. The owner accepted the
native identity/style mapping and subjective emotion criteria on 2026-11-15;
this does not introduce a V2ProPlus SV encoder into V5.
"""
from dataclasses import replace

from aniflive_tts.expression import ConditioningPolicy


def _component_reference(bank, value, attribute, expression_id):
    preferred = (bank.get(expression_id), bank.identity)
    for reference in (*preferred, *bank.references.values()):
        if getattr(reference, attribute) is value:
            return reference
    raise ValueError(f"Conditioning {attribute} does not belong to this voice")


def resolve_conditioning(bundle, bank):
    semantic = bundle.semantic
    if bundle.policy is ConditioningPolicy.FULL_SWITCH:
        return semantic
    style = _component_reference(bank, bundle.spectrogram, 'spectrogram', bundle.expression_id)
    identity = _component_reference(bank, bundle.speaker_embedding, 'speaker_embedding', bundle.expression_id)
    data = semantic.bert
    if not all(isinstance(ref.bert, dict) and len(ref.bert.get('reference', ())) == 7
               for ref in (semantic, style, identity)):
        raise ValueError('V5 conditioning requires prepared native reference components')
    sem, sty, voice = (ref.bert['reference'] for ref in (semantic, style, identity))
    composed = (*sem[:3], sty[3], voice[4], sty[5], voice[6])
    return replace(semantic, spectrogram=bundle.spectrogram,
                   speaker_embedding=bundle.speaker_embedding,
                   bert={**data, 'reference': composed, '_mixed_conditioning': True,
                         '_semantic_identity_id': id(semantic),
                         '_conditioning_key': (id(semantic), id(style), id(identity), bundle.policy.value),
                         '_conditioning_components': {'semantic': semantic.id, 'style': style.id,
                                                      'voice_anchor': identity.id,
                                                      'policy': bundle.policy.value}})
