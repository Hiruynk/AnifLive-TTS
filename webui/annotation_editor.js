(function (global) {
  "use strict";

  const SAFE_EXPRESSION_BOUNDARIES = new Set(Array.from(",.;?!、，。？！；"));
  const BOUNDARY_CLOSERS = new Set(Array.from("\"'”’」』）》】〕〉"));

  function hasSafeExpressionBoundary(text) {
    const source = String(text || "");
    if (/(?:\r?\n){2}\s*$/u.test(source)) return true;
    let ending = source.trimEnd();
    while (ending && BOUNDARY_CLOSERS.has(ending[ending.length - 1])) {
      ending = ending.slice(0, -1).trimEnd();
    }
    return Boolean(ending && SAFE_EXPRESSION_BOUNDARIES.has(ending[ending.length - 1]));
  }

  function isSafeExpressionRange(text, start, end) {
    const source = String(text || "");
    const prefix = source.slice(0, start);
    const selected = source.slice(start, end);
    const suffix = source.slice(end);
    const startsAtBoundary = !prefix.trim() || hasSafeExpressionBoundary(prefix);
    const endsAtBoundary = !suffix.trim() || hasSafeExpressionBoundary(selected);
    return startsAtBoundary && endsAtBoundary;
  }

  class AnnotationEditor {
    constructor(options) {
      this.editor = options.editor;
      this.textarea = options.textarea;
      this.mirror = options.mirror;
      this.mirrorContent = options.mirrorContent;
      this.captions = options.captions;
      this.menu = options.menu;
      this.cards = options.cards;
      this.translate = options.translate;
      this.profileLabel = options.profileLabel;
      this.profileColor = options.profileColor;
      this.resolvePrompt = options.resolvePrompt;
      this.languageName = options.languageName;
      this.onValidityChange = options.onValidityChange || (() => {});
      this.onStatus = options.onStatus || (() => {});
      this.maxAnnotations = options.maxAnnotations || 31;
      this.catalog = { enabled: false, profiles: [], policies: [] };
      this.language = "ja";
      this.annotations = [];
      this.playbackRange = null;
      this.playbackContent = document.createElement("div");
      this.playbackContent.className = "playback-mirror-content";
      this.mirrorContent.parentElement.append(this.playbackContent);
      this.pendingSelection = null;
      this.previousText = this.textarea.value;
      this.nextId = 1;
      this.annotationLineHeight = 2.5;
      this.resolveTimers = new Map();
      this.resizeObserver = new ResizeObserver(() => this.renderMirror());
      this.resizeObserver.observe(this.textarea);
      this.bindEvents();
      this.editor.classList.add("enhanced");
      this.renderAll();
    }

    bindEvents() {
      this.textarea.addEventListener("input", () => this.handleTextInput());
      this.textarea.addEventListener("scroll", () => this.syncScroll());
      this.textarea.addEventListener("mouseup", () => this.queueSelectionMenu());
      this.textarea.addEventListener("touchend", () => this.queueSelectionMenu());
      this.textarea.addEventListener("keyup", event => {
        if (event.shiftKey || ["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown"].includes(event.key)) {
          this.queueSelectionMenu(event.shiftKey);
        }
      });
      this.textarea.addEventListener("keydown", event => {
        if (event.key === "Escape") this.closeMenu();
      });
      this.menu.addEventListener("keydown", event => this.handleMenuKeydown(event));
      document.addEventListener("pointerdown", event => {
        if (!this.editor.contains(event.target)) this.closeMenu();
        if (!event.target?.closest?.(".expression-card-picker")) this.closeCardMenus();
      });
    }

    setCatalog(catalog, language) {
      this.catalog = catalog && typeof catalog === "object"
        ? catalog
        : { enabled: false, profiles: [], policies: [] };
      this.language = language;
      this.renderMenu();
      void this.revalidateAll();
    }

    setLanguage(language) {
      this.language = language;
      this.renderMenu();
      void this.revalidateAll();
    }

    setLocale() {
      for (const item of this.annotations) {
        const profile = item.resolved?.profile || item.profile;
        if (item.autoPrompt && profile) {
          item.prompt = this.profileLabel(profile);
        }
      }
      this.renderMenu();
      this.renderAll();
    }

    setDisabled(disabled) {
      this.textarea.disabled = disabled;
      if (disabled) {
        this.clearPlaybackRange();
        this.closeMenu();
        this.closeCardMenus();
      }
      this.cards.querySelectorAll("button").forEach(control => {
        control.disabled = disabled;
      });
    }

    setPlaybackRange(start, end) {
      const textLength = this.textarea.value.length;
      const next = Number.isInteger(start) && Number.isInteger(end)
        && start >= 0 && end > start && end <= textLength
        ? { start, end }
        : null;
      if (
        this.playbackRange?.start === next?.start
        && this.playbackRange?.end === next?.end
      ) return;
      this.playbackRange = next;
      this.renderPlayback();
    }

    clearPlaybackRange() {
      if (this.playbackRange === null) return;
      this.playbackRange = null;
      this.renderPlayback();
    }

    hasAnnotations() {
      return this.annotations.length > 0;
    }

    hasInvalidAnnotations() {
      this.refreshBoundaryErrors();
      return this.annotations.some(
        item => item.pending || item.error || item.boundaryError || !item.resolved
      );
    }

    annotationError(item) {
      return item.boundaryError || item.error || "";
    }

    refreshBoundaryErrors() {
      const text = this.textarea.value;
      for (const item of this.annotations) {
        item.boundaryError = isSafeExpressionRange(text, item.start, item.end)
          ? ""
          : this.translate("expressionUnsafeBoundary");
      }
    }

    buildSegments() {
      if (!this.annotations.length) return null;
      if (this.hasInvalidAnnotations()) throw new Error(this.translate("expressionInvalid"));
      const text = this.textarea.value;
      const ordered = this.sortedAnnotations();
      const segments = [];
      let cursor = 0;
      for (const item of ordered) {
        if (item.start > cursor) {
          segments.push({ text: text.slice(cursor, item.start), expression_prompt: null });
        }
        segments.push({ text: text.slice(item.start, item.end), expression_prompt: item.prompt });
        cursor = item.end;
      }
      if (cursor < text.length) {
        segments.push({ text: text.slice(cursor), expression_prompt: null });
      }
      const compact = segments.filter(item => item.text.length > 0);
      if (compact.map(item => item.text).join("") !== text) {
        throw new Error("Expression segment text no longer matches the editor text");
      }
      return compact;
    }

    sortedAnnotations() {
      return [...this.annotations].sort((left, right) => left.start - right.start || left.end - right.end);
    }

    profileRecord(profile) {
      const records = Array.isArray(this.catalog.profiles) ? this.catalog.profiles : [];
      return records.find(item => item && item.id === profile) || null;
    }

    supportsLanguage(profile) {
      return this.profileRecord(profile) !== null;
    }

    queueSelectionMenu(focusMenu = false) {
      global.setTimeout(() => this.openMenuFromSelection(focusMenu), 0);
    }

    openMenuFromSelection(focusMenu = false) {
      if (this.textarea.disabled || this.catalog.enabled !== true) {
        this.closeMenu();
        return;
      }
      let start = this.textarea.selectionStart;
      let end = this.textarea.selectionEnd;
      const value = this.textarea.value;
      while (start < end && /\s/u.test(value[start])) start += 1;
      while (end > start && /\s/u.test(value[end - 1])) end -= 1;
      if (start >= end) {
        this.closeMenu();
        return;
      }
      if (!isSafeExpressionRange(value, start, end)) {
        this.closeMenu();
        this.onStatus("expressionUnsafeBoundary", "error");
        return;
      }
      const overlaps = this.annotations.filter(item => item.start < end && item.end > start);
      if (this.annotations.length - overlaps.length >= this.maxAnnotations) {
        this.closeMenu();
        this.onStatus("expressionLimit", "error");
        return;
      }
      this.pendingSelection = { start, end };
      this.renderMenu();
      this.menu.hidden = false;
      global.requestAnimationFrame(() => {
        this.positionMenu(start, end);
        if (focusMenu) this.menu.querySelector(".expression-option")?.focus();
      });
    }

    positionMenu(start, end) {
      const rect = this.rangeRect(start, end) || this.textarea.getBoundingClientRect();
      const editorRect = this.editor.getBoundingClientRect();
      const menuRect = this.menu.getBoundingClientRect();
      const maximumLeft = Math.max(8, editorRect.width - menuRect.width - 8);
      const left = Math.min(maximumLeft, Math.max(8, rect.left - editorRect.left));
      const below = rect.bottom - editorRect.top + 7;
      const above = rect.top - editorRect.top - menuRect.height - 7;
      const viewportBottom = editorRect.top + below + menuRect.height;
      this.menu.style.left = `${left}px`;
      this.menu.style.top = `${viewportBottom < global.innerHeight - 8 || above < 8 ? below : above}px`;
    }

    closeMenu() {
      this.menu.hidden = true;
      this.pendingSelection = null;
    }

    renderMenu() {
      this.menu.replaceChildren();
      const records = Array.isArray(this.catalog.profiles) ? this.catalog.profiles : [];
      for (const record of records) {
        if (!record || typeof record.id !== "string" || record.id === "neutral") continue;
        this.menu.append(this.expressionOption(record.id));
      }
      const neutral = document.createElement("button");
      neutral.className = "expression-option expression-option-neutral";
      neutral.type = "button";
      neutral.role = "option";
      neutral.tabIndex = -1;
      const dot = document.createElement("span");
      dot.className = "expression-option-dot";
      dot.style.background = "transparent";
      dot.style.border = "1px solid var(--faint)";
      const label = document.createElement("span");
      label.textContent = `${this.profileLabel("neutral")} · ${this.translate("expressionNeutral").split("·").pop().trim()}`;
      neutral.append(dot, label);
      neutral.addEventListener("click", () => this.applyProfile(null));
      this.menu.append(neutral);
    }

    expressionOption(profile) {
      const option = document.createElement("button");
      option.className = "expression-option";
      option.type = "button";
      option.role = "option";
      option.tabIndex = -1;
      option.dataset.profile = profile;
      option.style.setProperty("--annotation-color", this.profileColor(profile));
      const dot = document.createElement("span");
      dot.className = "expression-option-dot";
      const label = document.createElement("span");
      label.textContent = this.profileLabel(profile);
      option.append(dot, label);
      option.addEventListener("click", () => this.applyProfile(profile));
      return option;
    }

    handleMenuKeydown(event) {
      const options = Array.from(this.menu.querySelectorAll(".expression-option"));
      const current = options.indexOf(document.activeElement);
      let target = -1;
      if (event.key === "ArrowDown" || event.key === "ArrowRight") target = (current + 1) % options.length;
      if (event.key === "ArrowUp" || event.key === "ArrowLeft") target = (current - 1 + options.length) % options.length;
      if (event.key === "Home") target = 0;
      if (event.key === "End") target = options.length - 1;
      if (target >= 0) {
        event.preventDefault();
        options[target]?.focus();
      } else if (event.key === "Escape") {
        event.preventDefault();
        this.closeMenu();
        this.textarea.focus();
      }
    }

    applyProfile(profile) {
      const selection = this.pendingSelection;
      if (!selection) return;
      if (profile && !isSafeExpressionRange(
        this.textarea.value, selection.start, selection.end
      )) {
        this.closeMenu();
        this.onStatus("expressionUnsafeBoundary", "error");
        return;
      }
      this.removeOverlapping(selection.start, selection.end);
      if (profile) {
        const item = {
          id: `expression-${this.nextId++}`,
          start: selection.start,
          end: selection.end,
          profile,
          prompt: this.profileLabel(profile),
          autoPrompt: true,
          resolved: null,
          pending: true,
          error: "",
          boundaryError: "",
          revision: 0,
          color: this.profileColor(profile)
        };
        this.annotations.push(item);
        this.annotations.sort((left, right) => left.start - right.start);
        this.closeMenu();
        this.renderAll();
        void this.resolveAnnotation(item.id);
      } else {
        this.closeMenu();
        this.renderAll();
      }
    }

    removeOverlapping(start, end) {
      const removed = this.annotations.filter(item => item.start < end && item.end > start);
      for (const item of removed) this.clearResolveTimer(item.id);
      this.annotations = this.annotations.filter(item => !(item.start < end && item.end > start));
    }

    removeAnnotation(id) {
      this.clearResolveTimer(id);
      this.annotations = this.annotations.filter(item => item.id !== id);
      this.renderAll();
    }

    clearResolveTimer(id) {
      const timer = this.resolveTimers.get(id);
      if (timer) global.clearTimeout(timer);
      this.resolveTimers.delete(id);
    }

    scheduleResolution(id) {
      this.clearResolveTimer(id);
      this.resolveTimers.set(id, global.setTimeout(() => {
        this.resolveTimers.delete(id);
        void this.resolveAnnotation(id);
      }, 250));
    }

    async resolveAnnotation(id) {
      const item = this.annotations.find(candidate => candidate.id === id);
      if (!item) return;
      const revision = ++item.revision;
      item.pending = true;
      item.error = "";
      this.updateCardState(item);
      this.notifyValidity();
      try {
        const resolved = await this.resolvePrompt(item.prompt);
        const current = this.annotations.find(candidate => candidate.id === id);
        if (!current || current.revision !== revision) return;
        if (!resolved || resolved.enabled !== true || !resolved.profile) {
          this.removeAnnotation(id);
          return;
        }
        current.resolved = resolved;
        current.profile = resolved.profile;
        current.color = this.profileColor(resolved.profile);
        current.pending = false;
        current.error = this.supportsLanguage(resolved.profile)
          ? ""
          : this.translate("expressionUnsupportedLanguage", {
              language: this.languageName(this.language)
            });
        this.renderAll();
      } catch (error) {
        const current = this.annotations.find(candidate => candidate.id === id);
        if (!current || current.revision !== revision) return;
        current.pending = false;
        current.resolved = null;
        current.error = String(error?.message || error || this.translate("expressionInvalid"));
        this.renderAll();
      }
    }

    async revalidateAll() {
      if (!this.annotations.length) {
        this.renderAll();
        return;
      }
      if (this.catalog.enabled !== true) {
        for (const item of this.annotations) {
          item.pending = false;
          item.resolved = null;
          item.error = this.translate("expressionInvalid");
        }
        this.renderAll();
        return;
      }
      await Promise.all(this.annotations.map(item => this.resolveAnnotation(item.id)));
    }

    handleTextInput() {
      const oldText = this.previousText;
      const newText = this.textarea.value;
      this.playbackRange = null;
      let prefix = 0;
      while (prefix < oldText.length && prefix < newText.length && oldText[prefix] === newText[prefix]) prefix += 1;
      let suffix = 0;
      while (
        suffix < oldText.length - prefix &&
        suffix < newText.length - prefix &&
        oldText[oldText.length - 1 - suffix] === newText[newText.length - 1 - suffix]
      ) suffix += 1;
      const oldEnd = oldText.length - suffix;
      const newEnd = newText.length - suffix;
      const delta = newEnd - oldEnd;
      const insertion = prefix === oldEnd;
      const kept = [];
      for (const item of this.sortedAnnotations()) {
        if (item.end <= prefix) {
          kept.push(item);
        } else if (item.start >= oldEnd) {
          kept.push({ ...item, start: item.start + delta, end: item.end + delta });
        } else if (
          (insertion && item.start < prefix && prefix < item.end) ||
          (!insertion && prefix >= item.start && oldEnd <= item.end)
        ) {
          const adjusted = { ...item, end: item.end + delta };
          if (adjusted.end > adjusted.start) kept.push(adjusted);
          else this.clearResolveTimer(item.id);
        } else {
          this.clearResolveTimer(item.id);
        }
      }
      this.annotations = kept;
      this.previousText = newText;
      this.closeMenu();
      this.renderAll();
    }

    renderAll() {
      this.refreshBoundaryErrors();
      this.renderMirror();
      this.renderCards();
      this.notifyValidity();
    }

    renderMirror() {
      const text = this.textarea.value;
      const fragment = document.createDocumentFragment();
      let cursor = 0;
      for (const item of this.sortedAnnotations()) {
        if (item.start > cursor) fragment.append(document.createTextNode(text.slice(cursor, item.start)));
        const span = document.createElement("span");
        const error = this.annotationError(item);
        span.className = `annotation-run${error ? " invalid" : ""}`;
        span.dataset.annotationId = item.id;
        span.style.setProperty("--annotation-color", error ? "var(--danger)" : item.color);
        span.textContent = text.slice(item.start, item.end);
        fragment.append(span);
        cursor = item.end;
      }
      if (cursor < text.length) fragment.append(document.createTextNode(text.slice(cursor)));
      if (!text) fragment.append(document.createTextNode(" "));
      this.mirrorContent.replaceChildren(fragment);
      this.renderPlayback();
      this.syncScroll();
      global.requestAnimationFrame(() => this.renderCaptions());
    }

    renderPlayback() {
      const text = this.textarea.value;
      const playback = this.playbackRange;
      if (!playback) {
        this.playbackContent.replaceChildren();
        return;
      }
      const fragment = document.createDocumentFragment();
      if (playback.start > 0) fragment.append(document.createTextNode(text.slice(0, playback.start)));
      const active = document.createElement("span");
      active.className = "playback-active-character";
      active.textContent = text.slice(playback.start, playback.end);
      fragment.append(active);
      if (playback.end < text.length) fragment.append(document.createTextNode(text.slice(playback.end)));
      this.playbackContent.replaceChildren(fragment);
      this.playbackContent.style.transform = this.mirrorContent.style.transform;
    }

    syncScroll() {
      const transform = `translate(${-this.textarea.scrollLeft}px, ${-this.textarea.scrollTop}px)`;
      this.mirrorContent.style.transform = transform;
      this.playbackContent.style.transform = transform;
      global.requestAnimationFrame(() => this.renderCaptions());
    }

    measureCaptionWidth(text) {
      if (!this.captionMeasureContext) {
        this.captionMeasureContext = document.createElement("canvas").getContext("2d");
      }
      const context = this.captionMeasureContext;
      if (!context) return Math.min(220, Math.max(24, Array.from(text).length * 10 + 6));
      const style = global.getComputedStyle(this.editor);
      context.font = `400 10px ${style.fontFamily}`;
      return Math.min(220, Math.max(24, Math.ceil(context.measureText(text).width) + 6));
    }

    renderCaptions() {
      this.captions.replaceChildren();
      const editorRect = this.editor.getBoundingClientRect();
      const candidates = [];
      for (const item of this.sortedAnnotations()) {
        const span = this.mirrorContent.querySelector(`[data-annotation-id="${item.id}"]`);
        if (!span) continue;
        const rects = Array.from(span.getClientRects()).filter(rect => rect.width > 0 && rect.height > 0);
        const rect = rects[rects.length - 1];
        if (!rect) continue;
        for (const lineRect of rects) {
          const hit = document.createElement("button");
          hit.className = "annotation-underline-hit";
          hit.type = "button";
          hit.tabIndex = -1;
          hit.setAttribute("aria-label", item.prompt);
          hit.style.left = `${Math.max(0, lineRect.left - editorRect.left)}px`;
          hit.style.top = `${lineRect.bottom - editorRect.top - 5}px`;
          hit.style.width = `${lineRect.width}px`;
          hit.addEventListener("click", () => this.focusCard(item.id));
          this.captions.append(hit);
        }
        const top = rect.bottom - editorRect.top + 1;
        if (top < 0 || top > editorRect.height) continue;
        const captionWidth = this.measureCaptionWidth(item.prompt);
        const centeredLeft = rect.left - editorRect.left + (rect.width - captionWidth) / 2;
        const maximumLeft = Math.max(7, editorRect.width - captionWidth - 7);
        candidates.push({
          item,
          rect,
          top,
          width: captionWidth,
          left: Math.min(maximumLeft, Math.max(7, centeredLeft)),
          lane: 0
        });
      }

      const lineGroups = [];
      let maximumLaneCount = 1;
      for (const candidate of candidates.sort((left, right) => (
        left.rect.top - right.rect.top || left.left - right.left
      ))) {
        let group = lineGroups.find(entry => Math.abs(entry.top - candidate.rect.top) < 3);
        if (!group) {
          group = { top: candidate.rect.top, lanes: [] };
          lineGroups.push(group);
        }
        let lane = group.lanes.findIndex(intervals => intervals.every(interval => (
          candidate.left + candidate.width <= interval.left ||
          candidate.left >= interval.right
        )));
        if (lane < 0) {
          lane = group.lanes.length;
          group.lanes.push([]);
        }
        group.lanes[lane].push({
          left: candidate.left,
          right: candidate.left + candidate.width
        });
        candidate.lane = lane;
        maximumLaneCount = Math.max(maximumLaneCount, lane + 1);
      }

      const requiredLineHeight = 2.5 + (maximumLaneCount - 1) * 1.2;
      if (Math.abs(requiredLineHeight - this.annotationLineHeight) > 0.01) {
        this.annotationLineHeight = requiredLineHeight;
        this.editor.style.setProperty("--annotation-line-height", String(requiredLineHeight));
        global.requestAnimationFrame(() => this.renderCaptions());
        return;
      }

      for (const candidate of candidates) {
        const { item } = candidate;
        const caption = document.createElement("button");
        caption.className = "annotation-caption";
        caption.type = "button";
        caption.dataset.annotationId = item.id;
        caption.textContent = item.prompt;
        caption.title = item.prompt;
        caption.style.setProperty(
          "--annotation-color", this.annotationError(item) ? "var(--danger)" : item.color
        );
        caption.style.left = `${candidate.left}px`;
        caption.style.top = `${candidate.top + candidate.lane * 17}px`;
        caption.style.width = `${candidate.width}px`;
        caption.addEventListener("click", () => this.focusCard(item.id));
        this.captions.append(caption);
      }
    }

    renderCards() {
      this.cards.replaceChildren();
      if (!this.annotations.length) {
        const empty = document.createElement("div");
        empty.className = "expression-empty";
        empty.textContent = this.catalog.enabled === true
          ? this.translate("expressionEmpty")
          : this.translate("expressionUnavailable");
        this.cards.append(empty);
        return;
      }
      for (const item of this.sortedAnnotations()) this.cards.append(this.expressionCard(item));
    }

    resizeProfileButton(button) {
      const label = button.querySelector(".expression-card-button-label");
      const textWidth = Array.from(label?.textContent || "").reduce(
        (width, character) => width + (character.codePointAt(0) > 0x7f ? 13 : 7.4),
        42
      );
      const availableWidth = Math.max(96, Math.min(420, this.cards.clientWidth - 48));
      button.style.width = `${Math.min(availableWidth, Math.max(86, textWidth))}px`;
    }

    closeCardMenus(except = null) {
      this.cards.querySelectorAll(".expression-card-picker").forEach(picker => {
        if (picker === except) return;
        const button = picker.querySelector(".expression-card-button");
        const menu = picker.querySelector(".expression-card-menu");
        if (button) button.setAttribute("aria-expanded", "false");
        if (menu) menu.hidden = true;
      });
    }

    positionCardMenu(picker, menu) {
      const pickerRect = picker.getBoundingClientRect();
      const cardsRect = this.cards.getBoundingClientRect();
      const menuWidth = Math.min(224, Math.max(180, pickerRect.width + 50, cardsRect.width - 12));
      menu.style.width = `${Math.min(menuWidth, cardsRect.width - 12)}px`;
      const measured = menu.getBoundingClientRect();
      const desiredLeft = (pickerRect.width - measured.width) / 2;
      const minimumLeft = cardsRect.left - pickerRect.left + 6;
      const maximumLeft = cardsRect.right - pickerRect.left - measured.width - 6;
      menu.style.left = `${Math.min(maximumLeft, Math.max(minimumLeft, desiredLeft))}px`;
    }

    openCardMenu(picker, button, menu, focusSelected = false) {
      const opening = menu.hidden;
      this.closeCardMenus(opening ? picker : null);
      menu.hidden = !opening;
      button.setAttribute("aria-expanded", String(opening));
      if (!opening) return;
      global.requestAnimationFrame(() => {
        this.positionCardMenu(picker, menu);
        if (focusSelected) {
          (menu.querySelector('[aria-selected="true"]') || menu.querySelector(".expression-card-menu-option"))?.focus();
        }
      });
    }

    selectCardProfile(id, profile) {
      const current = this.annotations.find(candidate => candidate.id === id);
      if (!current) return;
      current.profile = profile;
      current.prompt = this.profileLabel(profile);
      current.autoPrompt = true;
      current.color = this.profileColor(profile);
      current.pending = true;
      current.error = "";
      this.renderMirror();
      this.updateCardState(current);
      this.notifyValidity();
      void this.resolveAnnotation(current.id);
    }

    expressionCard(item) {
      const card = document.createElement("div");
      const error = this.annotationError(item);
      card.className = `expression-card${error ? " invalid" : ""}`;
      card.dataset.annotationId = item.id;
      card.style.setProperty("--annotation-color", error ? "var(--danger)" : item.color);
      const head = document.createElement("div");
      head.className = "expression-card-head";
      const snippet = document.createElement("span");
      snippet.className = "expression-card-text";
      snippet.textContent = this.textarea.value.slice(item.start, item.end);
      snippet.title = snippet.textContent;
      const remove = document.createElement("button");
      remove.className = "expression-remove";
      remove.type = "button";
      remove.textContent = "×";
      remove.title = this.translate("expressionRemove");
      remove.setAttribute("aria-label", this.translate("expressionRemove"));
      remove.disabled = this.textarea.disabled;
      remove.addEventListener("click", () => this.removeAnnotation(item.id));
      head.append(snippet);
      const picker = document.createElement("div");
      picker.className = "expression-card-picker";
      const pickerButton = document.createElement("button");
      pickerButton.className = "expression-card-button";
      pickerButton.type = "button";
      pickerButton.disabled = this.textarea.disabled;
      pickerButton.setAttribute("aria-haspopup", "listbox");
      pickerButton.setAttribute("aria-expanded", "false");
      pickerButton.setAttribute("aria-label", this.translate("expressionAria"));
      const buttonLabel = document.createElement("span");
      buttonLabel.className = "expression-card-button-label";
      const chevron = document.createElement("span");
      chevron.className = "expression-card-chevron";
      chevron.setAttribute("aria-hidden", "true");
      pickerButton.append(buttonLabel, chevron);
      const pickerMenu = document.createElement("div");
      pickerMenu.className = "expression-card-menu";
      pickerMenu.id = `expression-picker-${item.id}`;
      pickerMenu.role = "listbox";
      pickerMenu.hidden = true;
      pickerButton.setAttribute("aria-controls", pickerMenu.id);
      const records = Array.isArray(this.catalog.profiles) ? this.catalog.profiles : [];
      const selectedProfile = item.resolved?.profile || item.profile || "";
      for (const record of records) {
        if (!record || typeof record.id !== "string" || record.id === "neutral") continue;
        const option = document.createElement("button");
        option.className = "expression-card-menu-option";
        option.type = "button";
        option.role = "option";
        option.tabIndex = -1;
        option.dataset.profile = record.id;
        option.setAttribute("aria-selected", String(record.id === selectedProfile));
        option.style.setProperty("--option-color", this.profileColor(record.id));
        const swatch = document.createElement("span");
        swatch.className = "expression-card-menu-swatch";
        const optionLabel = document.createElement("span");
        optionLabel.textContent = this.profileLabel(record.id);
        option.append(swatch, optionLabel);
        option.addEventListener("click", () => {
          this.closeCardMenus();
          this.selectCardProfile(item.id, record.id);
        });
        pickerMenu.append(option);
      }
      if (selectedProfile && !pickerMenu.querySelector(`[data-profile="${selectedProfile}"]`)) {
        const unavailable = document.createElement("button");
        unavailable.className = "expression-card-menu-option";
        unavailable.type = "button";
        unavailable.role = "option";
        unavailable.tabIndex = -1;
        unavailable.dataset.profile = selectedProfile;
        unavailable.setAttribute("aria-selected", "true");
        unavailable.style.setProperty("--option-color", this.profileColor(selectedProfile));
        const swatch = document.createElement("span");
        swatch.className = "expression-card-menu-swatch";
        const optionLabel = document.createElement("span");
        optionLabel.textContent = this.profileLabel(selectedProfile);
        unavailable.append(swatch, optionLabel);
        pickerMenu.append(unavailable);
      }
      buttonLabel.textContent = this.profileLabel(selectedProfile);
      pickerButton.title = buttonLabel.textContent;
      pickerButton.addEventListener("click", () => this.openCardMenu(picker, pickerButton, pickerMenu));
      pickerButton.addEventListener("keydown", event => {
        if (["Enter", " ", "ArrowDown", "ArrowUp"].includes(event.key)) {
          event.preventDefault();
          if (pickerMenu.hidden) this.openCardMenu(picker, pickerButton, pickerMenu, true);
          else (pickerMenu.querySelector('[aria-selected="true"]') || pickerMenu.querySelector(".expression-card-menu-option"))?.focus();
        } else if (event.key === "Escape") {
          this.closeCardMenus();
        }
      });
      pickerMenu.addEventListener("keydown", event => {
        const options = Array.from(pickerMenu.querySelectorAll(".expression-card-menu-option"));
        const current = options.indexOf(document.activeElement);
        let target = -1;
        if (event.key === "ArrowDown") target = (current + 1) % options.length;
        if (event.key === "ArrowUp") target = (current - 1 + options.length) % options.length;
        if (event.key === "Home") target = 0;
        if (event.key === "End") target = options.length - 1;
        if (target >= 0) {
          event.preventDefault();
          options[target]?.focus();
        } else if (event.key === "Escape") {
          event.preventDefault();
          this.closeCardMenus();
          pickerButton.focus();
        }
      });
      picker.append(pickerButton, pickerMenu);
      const meta = document.createElement("div");
      meta.className = "expression-card-meta";
      meta.id = `expression-meta-${item.id}`;
      meta.dataset.expressionMeta = item.id;
      meta.setAttribute("aria-live", "polite");
      pickerButton.setAttribute("aria-describedby", meta.id);
      card.append(head, picker, remove, meta);
      card.addEventListener("click", event => {
        if (event.target === card || event.target === head || event.target === snippet) {
          this.focusRange(item.id);
        }
      });
      this.updateCardState(item, card);
      global.requestAnimationFrame(() => this.resizeProfileButton(pickerButton));
      return card;
    }

    updateCardState(item, suppliedCard = null) {
      const card = suppliedCard || this.cards.querySelector(`[data-annotation-id="${item.id}"]`);
      if (!card) return;
      const error = this.annotationError(item);
      card.classList.toggle("invalid", Boolean(error));
      card.setAttribute("aria-invalid", error ? "true" : "false");
      card.style.setProperty("--annotation-color", error ? "var(--danger)" : item.color);
      const dot = card.querySelector(".expression-card-dot");
      if (dot) dot.style.setProperty("--annotation-color", error ? "var(--danger)" : item.color);
      const meta = card.querySelector(`[data-expression-meta="${item.id}"]`);
      if (!meta) return;
      if (item.pending) meta.textContent = this.translate("expressionResolving");
      else if (error) {
        meta.textContent = error;
        meta.title = error;
      } else if (item.resolved) {
        meta.textContent = this.translate("expressionResolved", {
          profile: this.profileLabel(item.resolved.profile),
          intensity: Number(item.resolved.intensity).toFixed(2)
        });
        meta.title = meta.textContent;
      } else meta.textContent = this.translate("expressionInvalid");
      const snippet = this.textarea.value.slice(item.start, item.end);
      card.title = `${snippet} · ${meta.textContent}`;
    }

    notifyValidity() {
      this.onValidityChange(!this.hasInvalidAnnotations());
    }

    focusRange(id) {
      const item = this.annotations.find(candidate => candidate.id === id);
      if (!item) return;
      this.textarea.focus();
      this.textarea.setSelectionRange(item.start, item.end);
    }

    focusCard(id) {
      const button = this.cards.querySelector(`[data-annotation-id="${id}"] .expression-card-button`);
      button?.scrollIntoView({ block: "nearest", behavior: "smooth" });
      button?.focus();
    }

    rangeRect(start, end) {
      const walker = document.createTreeWalker(this.mirrorContent, NodeFilter.SHOW_TEXT);
      const nodes = [];
      let node;
      while ((node = walker.nextNode())) nodes.push(node);
      const locate = offset => {
        let consumed = 0;
        for (const textNode of nodes) {
          const length = textNode.nodeValue.length;
          if (offset <= consumed + length) return [textNode, Math.max(0, offset - consumed)];
          consumed += length;
        }
        const last = nodes[nodes.length - 1];
        return last ? [last, last.nodeValue.length] : null;
      };
      const startPoint = locate(start);
      const endPoint = locate(end);
      if (!startPoint || !endPoint) return null;
      const range = document.createRange();
      range.setStart(startPoint[0], startPoint[1]);
      range.setEnd(endPoint[0], endPoint[1]);
      const rects = Array.from(range.getClientRects()).filter(rect => rect.width > 0 && rect.height > 0);
      return rects[rects.length - 1] || range.getBoundingClientRect();
    }
  }

  global.AnifLiveTTSExpressionBoundaries = {
    hasSafeExpressionBoundary,
    isSafeExpressionRange
  };
  global.AnifLiveTTSAnnotationEditor = AnnotationEditor;
})(globalThis);
