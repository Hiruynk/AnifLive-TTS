
const{chromium}=require("playwright"),fs=require("fs"),assert=require("node:assert/strict");
(async()=>{const b=await chromium.launch();const c=await b.newContext({viewport:{width:1280,height:800},locale:"en-US",reducedMotion:"reduce"});const p=await c.newPage();const base="http://127.0.0.1:8000";
await p.addInitScript(()=>{caches.keys().then(names=>{if(!names.length)return caches.open("aniflive-tts-studio-shell-qa-old")})});
const r=await p.goto(base+"/",{waitUntil:"networkidle"});assert.equal(r.headers()["x-aniflive-ui-fixture"],"synthetic");await p.evaluate(()=>navigator.serviceWorker.ready);await p.reload({waitUntil:"networkidle"});
await p.locator("#studioLocaleButton").click();await p.locator('#studioLocaleMenu [data-locale="zh-Hant"]').click();
const storage=await c.storageState();const localKeys=storage.origins.flatMap(o=>o.localStorage.map(v=>v.name));assert.deepEqual([...new Set(localKeys)],["aniflive.uiLocale"]);
const cached=await p.evaluate(async()=>{const names=await caches.keys();const result=[];for(const name of names){const cache=await caches.open(name);result.push({name,urls:(await cache.keys()).map(r=>new URL(r.url).pathname)})}return result});
const swSource=await (await p.request.get(base+"/studio-sw.js")).text();const expectedCache=swSource.match(/const CACHE_NAME = "([^"]+)"/)[1];assert.equal(cached.length,1);assert.equal(cached[0].name,expectedCache);assert.ok(cached[0].urls.length>10);assert.ok(cached[0].urls.every(u=>!/^\/(api|v1|media)\//.test(u)));
const offlineDocument=await p.evaluate(async()=>{const response=await caches.match("/offline.html");return response?.text()});
assert.match(offlineDocument,/Studio is offline/);
await p.goto(base+"/offline.html");assert.match(await p.locator("h1").innerText(),/offline/i);
await c.setOffline(false);await p.goto(base+"/",{waitUntil:"networkidle"});await p.locator("body.studio-ux").waitFor();
const result={cache:cached[0].name,staticAssets:cached[0].urls.length,onlyLocaleStored:true,oldCacheRemoved:true,cachedOfflineDocumentAvailable:true,onlineReturn:true};
fs.writeFileSync("/qa/editorial-pwa-results.json",JSON.stringify(result,null,2));console.log(result);await b.close()})().catch(e=>{console.error(e);process.exit(1)});
