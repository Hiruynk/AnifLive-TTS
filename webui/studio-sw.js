"use strict";

const CACHE_NAME = "aniflive-tts-studio-shell-1.4.0-editorial20";
const OFFLINE_SHELL = "/offline.html";
const STATIC_ASSETS = new Set([
  OFFLINE_SHELL,
  "/assets/studio.css",
  "/assets/studio.js",
  "/assets/studio_i18n.js",
  "/assets/styled_select.css",
  "/assets/styled_select.js",
  "/assets/job_controls.js",
  "/assets/dataset_factory_model.js",
  "/assets/tse_workstation_model.js",
  "/assets/lucide.min.js",
  "/pwa/studio-icon.svg",
  "/pwa/studio-icon-192.png",
  "/pwa/studio-icon-512.png"
]);

self.addEventListener("install", event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll([...STATIC_ASSETS]))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", event => {
  event.waitUntil(
    caches.keys()
      .then(keys => Promise.all(keys.filter(key => key !== CACHE_NAME).map(key => caches.delete(key))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", event => {
  const request = event.request;
  if (request.method !== "GET" || request.headers.has("range")) return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/v1/") || url.pathname.startsWith("/media/")) return;

  if (request.mode === "navigate") {
    event.respondWith(fetch(request).catch(() => caches.match(OFFLINE_SHELL)));
    return;
  }
  if (!STATIC_ASSETS.has(url.pathname)) return;
  event.respondWith(caches.match(request).then(cached => cached || fetch(request)));
});
