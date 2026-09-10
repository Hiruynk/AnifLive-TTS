(() => {
  "use strict";

  const instances = new Map();
  let openInstance = null;
  let sequence = 0;

  function currentOption(select) {
    return select.options[select.selectedIndex] || select.options[0] || null;
  }

  function close(instance, { focus = false } = {}) {
    if (!instance || instance.menu.hidden) return;
    instance.menu.hidden = true;
    instance.wrapper.classList.remove("is-open");
    instance.trigger.setAttribute("aria-expanded", "false");
    instance.activeIndex = -1;
    if (openInstance === instance) openInstance = null;
    if (focus) instance.trigger.focus();
  }

  function position(instance) {
    if (instance.menu.hidden) return;
    const rect = instance.trigger.getBoundingClientRect();
    const viewportGap = 10;
    const menuGap = 6;
    const visualViewport = window.visualViewport;
    const viewport = {
      left: visualViewport?.offsetLeft || 0,
      top: visualViewport?.offsetTop || 0,
      right: (visualViewport?.offsetLeft || 0) + (visualViewport?.width || document.documentElement.clientWidth),
      bottom: (visualViewport?.offsetTop || 0) + (visualViewport?.height || window.innerHeight),
    };
    const dialog = instance.menu.parentElement instanceof HTMLDialogElement ? instance.menu.parentElement : null;
    const dialogRect = dialog?.getBoundingClientRect();
    const bounds = {
      left: Math.max(viewport.left + viewportGap, dialogRect ? dialogRect.left + viewportGap : -Infinity),
      top: Math.max(viewport.top + viewportGap, dialogRect ? dialogRect.top + viewportGap : -Infinity),
      right: Math.min(viewport.right - viewportGap, dialogRect ? dialogRect.right - viewportGap : Infinity),
      bottom: Math.min(viewport.bottom - viewportGap, dialogRect ? dialogRect.bottom - viewportGap : Infinity),
    };
    const boundedWidth = Math.max(1, bounds.right - bounds.left);
    const width = Math.min(Math.max(rect.width, Math.min(180, boundedWidth)), boundedWidth);
    const left = Math.min(Math.max(bounds.left, rect.left), bounds.right - width);
    const availableBelow = Math.max(0, bounds.bottom - rect.bottom - menuGap);
    const availableAbove = Math.max(0, rect.top - bounds.top - menuGap);
    const naturalHeight = Math.min(310, instance.menu.scrollHeight || 310);
    const openAbove = naturalHeight > availableBelow && availableAbove > availableBelow;
    const availableHeight = openAbove ? availableAbove : availableBelow;
    const maxHeight = Math.max(1, Math.min(naturalHeight, availableHeight || bounds.bottom - bounds.top));
    instance.menu.style.width = `${width}px`;
    instance.menu.style.maxHeight = `${maxHeight}px`;
    const renderedHeight = Math.min(instance.menu.offsetHeight, maxHeight);
    const top = Math.min(
      Math.max(bounds.top, openAbove ? rect.top - renderedHeight - menuGap : rect.bottom + menuGap),
      bounds.bottom - renderedHeight,
    );
    const origin = dialog?.getBoundingClientRect() || { left: 0, top: 0 };
    const scrollLeft = dialog?.scrollLeft || 0;
    const scrollTop = dialog?.scrollTop || 0;
    instance.menu.style.position = dialog ? "absolute" : "fixed";
    instance.menu.style.left = `${left - origin.left + scrollLeft}px`;
    instance.menu.style.top = `${top - origin.top + scrollTop}px`;
    instance.menu.style.transformOrigin = openAbove ? "bottom center" : "top center";
    instance.menu.dataset.placement = openAbove ? "top" : "bottom";
  }

  function setActive(instance, index, { focus = true } = {}) {
    const buttons = [...instance.menu.querySelectorAll(".styled-select-option:not(:disabled)")];
    if (!buttons.length) return;
    const next = Math.max(0, Math.min(buttons.length - 1, index));
    buttons.forEach(button => button.classList.remove("is-active"));
    buttons[next].classList.add("is-active");
    instance.activeIndex = next;
    if (focus) buttons[next].focus({ preventScroll: true });
    buttons[next].scrollIntoView({ block: "nearest" });
  }

  function choose(instance, option) {
    if (!option || option.disabled) return;
    const changed = instance.select.value !== option.value;
    instance.select.value = option.value;
    window.setTimeout(() => {
      close(instance, { focus: true });
      sync(instance);
      if (changed) instance.select.dispatchEvent(new Event("change", { bubbles: true }));
    }, 0);
  }

  function renderOptions(instance) {
    const selected = currentOption(instance.select);
    instance.menu.replaceChildren();
    [...instance.select.options].forEach((option, index) => {
      const button = document.createElement("button");
      const label = document.createElement("span");
      button.type = "button";
      button.className = "styled-select-option";
      button.id = `${instance.id}-option-${index}`;
      button.setAttribute("role", "option");
      button.setAttribute("aria-selected", option === selected ? "true" : "false");
      button.dataset.optionIndex = String(index);
      button.disabled = option.disabled;
      label.textContent = option.textContent;
      button.append(label);
      button.addEventListener("click", event => {
        event.preventDefault();
        event.stopPropagation();
        choose(instance, option);
      });
      button.addEventListener("pointermove", () => {
        const enabled = [...instance.menu.querySelectorAll(".styled-select-option:not(:disabled)")];
        setActive(instance, enabled.indexOf(button), { focus: false });
      });
      instance.menu.append(button);
    });
  }

  function sync(instance, opening = false) {
    const option = currentOption(instance.select);
    instance.value.textContent = option?.textContent || "";
    instance.trigger.disabled = instance.select.disabled;
    // Studio may contain thousands of evidence options. Build the popup when opened.
    if (!document.body.classList.contains("studio-ux")) {
      renderOptions(instance);
      return;
    }
    if (instance.menu.hidden && !opening) return;
    const focused = instance.menu.contains(document.activeElement);
    const focusedIndex = Number(document.activeElement?.dataset?.optionIndex);
    const focusedValue = focused ? instance.select.options[focusedIndex]?.value : null;
    const scrollTop = instance.menu.scrollTop;
    renderOptions(instance);
    if (focused) {
      const enabled = [...instance.menu.querySelectorAll(".styled-select-option:not(:disabled)")];
      const index = enabled.findIndex(button => instance.select.options[Number(button.dataset.optionIndex)]?.value === focusedValue);
      if (index >= 0) setActive(instance, index);
    }
    if (!instance.menu.hidden) instance.menu.scrollTop = scrollTop;
  }

  function open(instance) {
    if (instance.select.disabled) return;
    if (openInstance && openInstance !== instance) close(openInstance);
    sync(instance, true);
    instance.menu.hidden = false;
    instance.wrapper.classList.add("is-open");
    instance.trigger.setAttribute("aria-expanded", "true");
    openInstance = instance;
    position(instance);
    const selected = [...instance.menu.querySelectorAll(".styled-select-option:not(:disabled)")]
      .findIndex(button => button.getAttribute("aria-selected") === "true");
    setActive(instance, selected < 0 ? 0 : selected);
    window.requestAnimationFrame(() => position(instance));
  }

  function handleKeydown(instance, event) {
    const enabled = [...instance.menu.querySelectorAll(".styled-select-option:not(:disabled)")];
    if (event.key === "Escape") {
      // Let a containing dialog handle Escape when this select is already closed.
      if (instance.menu.hidden) return;
      event.preventDefault();
      close(instance, { focus: true });
      return;
    }
    if (event.key === "Tab") {
      close(instance);
      return;
    }
    if (["ArrowDown", "ArrowUp", "Home", "End"].includes(event.key)) {
      event.preventDefault();
      if (instance.menu.hidden) {
        open(instance);
        return;
      }
      const delta = event.key === "ArrowDown" ? 1 : event.key === "ArrowUp" ? -1 : 0;
      const index = event.key === "Home" ? 0 : event.key === "End" ? enabled.length - 1 : instance.activeIndex + delta;
      setActive(instance, index);
      return;
    }
    if ((event.key === "Enter" || event.key === " ") && instance.menu.hidden) {
      event.preventDefault();
      open(instance);
      return;
    }
    if (event.key === "Enter" || event.key === " ") {
      const optionButton = document.activeElement?.closest?.(".styled-select-option");
      if (optionButton) {
        event.preventDefault();
        const option = instance.select.options[Number(optionButton.dataset.optionIndex)];
        choose(instance, option);
      }
    }
  }

  function enhance(select) {
    if (!(select instanceof HTMLSelectElement) || select.multiple || select.size > 1 || instances.has(select)) return null;
    const id = `styled-select-${++sequence}`;
    const wrapper = document.createElement("div");
    const trigger = document.createElement("button");
    const value = document.createElement("span");
    const chevron = document.createElement("span");
    const menu = document.createElement("div");
    wrapper.className = "styled-select";
    trigger.type = "button";
    trigger.className = "styled-select-trigger";
    trigger.setAttribute("role", "combobox");
    trigger.setAttribute("aria-haspopup", "listbox");
    trigger.setAttribute("aria-expanded", "false");
    trigger.setAttribute("aria-controls", `${id}-listbox`);
    if (select.hasAttribute("aria-label")) trigger.setAttribute("aria-label", select.getAttribute("aria-label"));
    value.className = "styled-select-value";
    chevron.className = "styled-select-chevron";
    chevron.setAttribute("aria-hidden", "true");
    menu.id = `${id}-listbox`;
    menu.className = "styled-select-listbox";
    menu.setAttribute("role", "listbox");
    menu.hidden = true;
    select.parentNode.insertBefore(wrapper, select);
    wrapper.append(select, trigger);
    trigger.append(value, chevron);
    (select.closest("dialog") || document.body).append(menu);
    select.classList.add("styled-select-native");
    select.tabIndex = -1;
    select.setAttribute("aria-hidden", "true");
    const instance = { id, select, wrapper, trigger, value, menu, activeIndex: -1, observer: null };
    instances.set(select, instance);
    trigger.addEventListener("click", () => instance.menu.hidden ? open(instance) : close(instance));
    trigger.addEventListener("keydown", event => handleKeydown(instance, event));
    menu.addEventListener("keydown", event => handleKeydown(instance, event));
    select.addEventListener("focus", () => trigger.focus());
    select.addEventListener("change", () => sync(instance));
    instance.observer = new MutationObserver(() => sync(instance));
    instance.observer.observe(select, { childList: true, subtree: true, attributes: true });
    sync(instance);
    return instance;
  }

  function scan(root = document) {
    if (root instanceof HTMLSelectElement) enhance(root);
    root.querySelectorAll?.("select").forEach(enhance);
  }

  function release(root) {
    const candidates = [];
    if (root instanceof HTMLSelectElement) candidates.push(root);
    root.querySelectorAll?.("select").forEach(select => candidates.push(select));
    queueMicrotask(() => candidates.forEach(select => {
      if (select.isConnected) return;
      const instance = instances.get(select);
      if (!instance) return;
      if (openInstance === instance) openInstance = null;
      instance.observer.disconnect();
      instance.menu.remove();
      instances.delete(select);
    }));
  }

  function refresh(target) {
    if (target instanceof HTMLSelectElement) {
      const instance = instances.get(target);
      if (instance) sync(instance);
      else enhance(target);
      return;
    }
    instances.forEach(sync);
  }

  document.addEventListener("pointerdown", event => {
    if (!openInstance) return;
    if (!openInstance.wrapper.contains(event.target) && !openInstance.menu.contains(event.target)) close(openInstance);
  }, true);
  window.addEventListener("resize", () => openInstance && position(openInstance));
  window.addEventListener("scroll", () => openInstance && position(openInstance), true);
  window.visualViewport?.addEventListener("resize", () => openInstance && position(openInstance));
  window.visualViewport?.addEventListener("scroll", () => openInstance && position(openInstance));
  new MutationObserver(mutations => {
    mutations.forEach(mutation => {
      mutation.addedNodes.forEach(node => {
        if (node instanceof Element) scan(node);
      });
      mutation.removedNodes.forEach(node => {
        if (node instanceof Element) release(node);
      });
    });
  }).observe(document.documentElement, { childList: true, subtree: true });

  window.AnifLiveTTSStyledSelect = Object.freeze({ enhance, refresh, refreshAll: () => refresh() });
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => scan());
  else scan();
})();
