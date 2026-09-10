# AnifLive-TTS v1.4 WebUI Design

## Product Surface

The interface is a local voice production workstation. Its Overview uses an
editorial motion scene that fills the browser viewport while keeping live
operational state and active work in the first viewport. The video is ambient
media rather than a framed card or a glass surface.

Every Studio product lockup uses the full name **AnifLive-TTS Studio**. The
preserved lightweight synthesis surface is named **AnifLive-TTS WebUI**.

## Visual System

- Pixzen informs the monochrome editorial hierarchy, whitespace and large
  first-view product typography.
- MotionSite Outbox informs the job dependency spine and one dominant active
  workflow.
- MotionSite Features Analytics informs asymmetric evaluation and runtime
  layouts.
- MotionSite Features Kinetic informs staggered information weight without a
  uniform card wall.
- Ostra informs only elevated inspectors, dialogs and transports: layered
  translucency, background refraction, a lit edge and restrained depth.
- Echo Studio informs the wine and lilac synthesis atmosphere.
- SpaceUp is reserved for the Overview and true empty states. It is not a
  persistent dashboard background or a module-transition overlay.

The source references guide hierarchy and motion only. Their hospitality,
marketing and decorative compositions are not copied. The Overview uses the
Mixkit Stock Video Free License asset documented in `webui/media/README.md`.
Liquid glass is an interface material, never the background video.

## Layout

Desktop uses a 72 px Lucide icon dock, a 56 px utility strip and a 12-column
unframed content canvas. The Overview uses a 108 px product title over
full-viewport motion; module pages use a 62 px editorial title and a fixed
action zone. Inspectors are elevated only when they hold a real selection or
control surface.

The Overview title treats `AnifLive-TTS` as one non-breaking unit. At narrower
sizes only `Studio` may move to the next line; the full product name remains
intact.

Mobile uses a 52 px utility strip and five bottom actions: Overview, Synthesis,
Datasets, Jobs and More. Secondary modules open in a bottom sheet. Tables become
labelled rows and inspectors move above the primary canvas.

## Color And Material

- Background: `#0B090D`
- Base surface: `#121015`
- Tool surface: `#17131A`
- Hairline: `#342C36`
- Text: `#F4F0F3`
- Wine: `#8F3157`
- Rose: `#C35F82`
- Lilac: `#A98BD0`

Gold remains exclusive to playback-follow text and critical audio focus. Green
and cyan are reserved for runtime state. Page sections remain matte and
unframed. The Voice Pulse transport is the primary liquid-glass moment;
inspectors and dialogs may use restrained depth when their interaction requires
it, while the surrounding workflow canvas stays flat.

## Motion

Module changes never introduce a standalone card, interstitial or full-screen
transition layer. The destination canvas fades and moves into place while its
title reveals inline; the module itself remains visible throughout the motion.
Hover motion is limited to one pixel. The Overview Voice Pulse animates while a
workstation job or Synthesis playback is active. Background video pauses outside
Overview, and all motion honors `prefers-reduced-motion`.

## Synthesis Contract

The v1.3 Synthesis implementation remains the interaction baseline. v1.4
preserves segmented expression underlines and captions, expression tags,
language and model selection, streaming playback, audio-aligned gold text,
cancellation, downloads, eight metrics and the five-request history.
