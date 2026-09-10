from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_workstation_uses_vetted_local_motion_asset() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    source = (ROOT / "webui" / "media" / "README.md").read_text(encoding="utf-8")

    assert index.count("/media/voice-workstation-background.mp4") == 1
    assert index.count("/media/voice-workstation-poster.jpg") == 1
    assert '<div class="ambient-stage" aria-hidden="true">' in index
    assert '<video id="overviewFilm" autoplay muted loop playsinline' in index
    assert "voice-pulse-master" not in index
    assert "voice-pulse-loop" not in index
    assert "mixkit.co/free-stock-video/abstract-background-in-grayscale-101010/" in source
    assert "Mixkit Stock Video Free License" in source
    assert "licenses/MIXKIT-STOCK-VIDEO-FREE-LICENSE.txt" in source

    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    assert "*.mp4" in dockerignore
    assert "!webui/media/voice-workstation-background.mp4" in dockerignore


def test_overview_motion_is_full_viewport_and_glass_is_selective() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")

    assert ".ambient-stage {\n  position: fixed;\n  inset: 0;" in css
    assert ".ambient-stage video,\n.ambient-stage-shade" in css
    assert "object-fit: cover;" in css
    assert "filter: grayscale(.12) saturate(.68) contrast(1.08) brightness(.54);" in css
    assert "background: rgba(8, 6, 10, .42);" in css
    assert ".overview-view {" in css
    assert "background: transparent;" in css
    assert '<section class="pulse-field voice-pulse-glass"' not in index
    assert 'id="voicePulse"' not in index
    assert 'id="pulseState"' not in index


def test_global_locale_picker_and_workstation_layout_contract() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")
    i18n = (ROOT / "webui" / "studio_i18n.js").read_text(encoding="utf-8")

    assert 'id="refreshButton"' not in index
    assert 'id="studioLocaleButton"' in index
    assert 'data-lucide="globe-2"' in index
    assert 'data-locale="en"' in index
    assert 'data-locale="zh-Hant"' in index
    assert 'data-locale="zh-Hans"' in index
    assert 'class="rail-footer"' not in index
    assert '>01</span> CREATE<' in index
    assert '<h1>Speech Synthesis</h1>' in index
    assert 'id="tseAudioPlayer"' in index
    assert 'id="tseAudioToggle"' in index
    assert 'id="tseAudioSeek"' in index
    assert 'id="tseAudioMute"' in index
    assert ".studio-locale-menu {" in css
    assert ".studio-audio-player {" in css
    assert "border-right: 1px solid var(--line-soft);" in css
    assert "border-bottom: 1px solid var(--line-soft);" in css
    assert ".training-run-monitor .run-metrics > div:last-child { grid-column: span 2; }" in css
    assert 'const LOCALE_KEY = "aniflive.uiLocale"' in i18n
    assert '"zh-Hant"' in i18n
    assert '"zh-Hans"' in i18n
    assert '"No dataset projects": ["尚未建立資料集專案", "尚未创建数据集项目"]' in i18n
    assert '"Capability status unavailable": ["尚未取得能力狀態", "尚未取得能力状态"]' in i18n


def test_embedded_synthesis_uses_the_studio_page_scrollport() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")
    studio = (ROOT / "webui" / "studio.js").read_text(encoding="utf-8")
    synthesis = (ROOT / "webui" / "synthesis.html").read_text(encoding="utf-8")

    assert 'id="synthesisFrame"' in index
    assert 'scrolling="no"' in index
    assert 'type: "aniflive-tts:synthesis-height"' in synthesis
    assert 'event.data?.type === "aniflive-tts:synthesis-height"' in studio
    assert 'frame.style.height = `${height}px`' in studio
    assert ".synthesis-view {\n  height: auto;\n  min-height: 100%;" in css
    assert "padding: 32px 38px 0;\n  overflow: visible;" in css


def test_module_navigation_has_no_transition_card_or_overlay() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")
    javascript = (ROOT / "webui" / "studio.js").read_text(encoding="utf-8")

    for text in (index, css, javascript):
        assert "module-transition" not in text
        assert "moduleTransition" not in text

    assert ".view.active {\n  animation: canvas-enter" in css
    assert "animation: title-reveal" in css
    assert "@keyframes canvas-enter" in css
    assert "@keyframes title-reveal" in css
    assert "clip-path" not in css


def test_editorial_headings_preserve_descenders_at_every_breakpoint() -> None:
    css = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")

    assert ".module-header h1 {" in css
    assert "padding-bottom: .14em;" in css
    assert css.count("line-height: 1.14;") >= 2
    assert "line-height: .94;" not in css
    assert "line-height: .98;" not in css


def test_mobile_workstation_tables_and_long_labels_remain_reachable() -> None:
    css = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")
    synthesis = (ROOT / "webui" / "synthesis.html").read_text(encoding="utf-8")
    editor = (ROOT / "webui" / "annotation_editor.js").read_text(encoding="utf-8")

    assert ".dataset-items-table,\n  .tse-segment-wrap table,\n  .evaluation-language-table" in css
    assert ".section-heading {\n    flex-wrap: wrap;" in css
    assert "min-width: 840px;" in css
    assert "overflow-wrap: anywhere;" in synthesis
    assert ".expression-card-picker {" in synthesis
    assert "max-width: calc(100% - 30px);" in synthesis
    assert "resizeProfileButton" not in editor


def test_medium_workstation_width_preserves_gpu_identity() -> None:
    css = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")

    assert ".runtime-strip > div:first-child { min-width: 190px; max-width: 210px; }" in css
    assert ".runtime-strip > div:nth-child(2) { display: none; }" in css
    assert ".runtime-strip > div.studio-locale-picker { display: block; }" in css
    assert "#overviewGpu {" in css
    assert "text-overflow: clip;" in css
    assert "white-space: normal;" in css


def test_model_identity_is_scoped_to_synthesis_and_overview_guides_creation() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "webui" / "studio.js").read_text(encoding="utf-8")
    i18n = (ROOT / "webui" / "studio_i18n.js").read_text(encoding="utf-8")
    style = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")

    assert 'id="runtimeModel"' not in index
    assert 'id="overviewModel"' not in index
    assert "ACTIVE MODEL" not in index
    assert 'id="overviewStartModelButton"' in index
    assert "Start building a voice model" in index
    assert 'byId("overviewStartModelButton").addEventListener("click"' in script
    assert 'destination?.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" })' in script
    assert 'newProject?.focus({ preventScroll: true })' in script
    assert '"Start building a voice model": ["開始建立語音模型", "开始创建语音模型"]' in i18n
    assert ".recent-work.is-guiding .section-heading::after" in style
    assert ".metric-baseline { grid-template-columns: repeat(3, minmax(0, 1fr)); }" in style


def test_dense_surface_alignment_guards_are_explicit() -> None:
    style = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")

    assert ".component-bundle-actions > button {" in style
    assert "margin-top: 22px;" in style
    assert ".inspector .dataset-summary > div" in style
    assert "grid-template-columns: minmax(0, 1fr);" in style
    assert "font-size: clamp(16px, 1.6vw, 21px);" in style
    assert ".domain-workspace,\n.jobs-layout { grid-template-columns: minmax(0, 1fr) 360px; }" in style


def test_studio_uses_styled_validation_instead_of_native_error_bubbles() -> None:
    script = (ROOT / "webui" / "studio.js").read_text(encoding="utf-8")
    style = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")

    assert "function installStyledValidation()" in script
    assert "form.noValidate = true" in script
    assert 'document.addEventListener("invalid"' in script
    assert 'event.stopImmediatePropagation()' in script
    assert ".field-validation-message" in style
    assert ".form-error:not(:empty)" in style
    assert "alert(" not in script
    assert "confirm(" not in script


def test_model_registry_and_evidence_have_independent_scrollports() -> None:
    style = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")

    assert 'body[data-current-view="models"] .viewport { overflow-y: hidden; }' in style
    assert ".registry-layout > .table-section > .table-wrap" in style
    assert ".registry-layout > .artifact-inspector" in style
    assert style.count("overscroll-behavior: contain;") >= 2
    assert style.count("scrollbar-gutter: stable;") >= 2


def test_path_pickers_center_empty_copy_and_constrain_narrow_labels() -> None:
    script = (ROOT / "webui" / "studio.js").read_text(encoding="utf-8")
    style = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")

    assert "function syncPathPickerEmptyState(input)" in script
    assert '.classList.toggle("is-empty", !input.value.trim())' in script
    assert 'create("span", "path-picker-placeholder", input.placeholder)' in script
    assert ".path-picker-control { align-items: center; }" in style
    assert ".studio-path-picker.is-empty textarea" in style
    assert ".studio-path-picker.is-empty .path-picker-placeholder { display: block; }" in style
    assert ".path-picker-browse {" in style
    assert "align-self: stretch;" in style
    assert "font-size: 10px;" in style


def test_jobs_page_has_independent_queue_and_event_scrollports() -> None:
    style = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")

    assert 'body[data-current-view="jobs"] .viewport { overflow-y: hidden; }' in style
    assert ".jobs-layout > .table-section > .job-table-wrap" in style
    assert ".jobs-layout > .queue-depth" in style
    assert ".view[data-view-panel=\"jobs\"] .jobs-layout" in style


def test_settings_history_is_a_compact_action_band() -> None:
    style = (ROOT / "webui" / "studio.css").read_text(encoding="utf-8")

    assert "min-height: 132px;" in style
    assert "grid-template-columns: minmax(0, 1fr) minmax(210px, 300px);" in style
    assert ".settings-history-action .secondary-button { width: 100%; min-height: 40px; }" in style


def test_settings_omits_the_redundant_committed_execution_panel() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")

    assert 'class="workspace-section settings-section settings-session"' not in index
    assert 'id="settingsSessionContext"' not in index
    assert "Committed execution" not in index


def test_public_webui_renders_user_content_without_inner_html() -> None:
    for relative in (
        "webui/index.html",
        "webui/studio.js",
        "webui/job_controls.js",
        "webui/synthesis.html",
        "webui/annotation_editor.js",
    ):
        text = (ROOT / relative).read_text(encoding="utf-8")
        assert "innerHTML" not in text, relative


def test_visible_product_lockups_use_full_product_name() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")

    bare_product = ">" + "Ani" + "fLive" + "<"
    assert bare_product not in index
    assert "AnifLive-TTS Studio" in index


def test_studio_uses_only_bundled_expression_analysis_icon() -> None:
    index = (ROOT / "webui" / "index.html").read_text(encoding="utf-8")

    assert 'data-lucide="scan-waveform"' not in index
    assert 'id="expressionAnalyzeButton"' in index
    assert 'data-lucide="audio-lines"' in index
