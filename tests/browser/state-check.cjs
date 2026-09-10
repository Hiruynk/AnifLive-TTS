
const {chromium}=require("playwright");const fs=require("fs");const assert=require("node:assert/strict");
(async()=>{
 const browser=await chromium.launch({headless:true,args:["--disable-gpu"]});const results=[];
 const page=await browser.newPage({viewport:{width:390,height:844},hasTouch:true,isMobile:true,reducedMotion:"reduce",locale:"en-US"});
 const response=await page.goto((process.env.STUDIO_QA_BASE || "http://host.docker.internal:9893") + "/",{waitUntil:"networkidle"});
 assert.equal(response.headers()["x-aniflive-ui-fixture"],"synthetic");
 await page.route("**/api/workstation/projects",r=>r.fulfill({json:{data:[]}}));
 await page.route("**/api/workstation/jobs",r=>r.fulfill({json:{data:[]}}));
 await page.reload({waitUntil:"networkidle"});
 await page.evaluate(()=>location.hash="training");
 await page.locator('[data-view-panel="training"].active').waitFor();
 assert.ok(await page.locator("#trainingRows").innerText());
 assert.equal(await page.locator("#trainingRunButton").isDisabled(),true);
 results.push("Empty project state keeps required actions disabled and explains no selection");
 await page.unroute("**/api/workstation/jobs");
 const jobs=Array.from({length:150},(_,i)=>({id:"job_ui_fixture_"+i,type:"engine.prepare",status:i%3===0?"failed":i%3===1?"queued":"succeeded",progress:i%3===2?1:0,parameters:{},depends_on:[],created_at:"2026-09-08T00:00:00Z",updated_at:"2026-09-08T00:00:00Z",wait_reason:i%3===1?"Waiting for resources · 測試用長原因說明，保持完整可讀":null,error:i%3===0?"Synthetic test failure":null}));
 await page.route("**/api/workstation/jobs",r=>r.fulfill({json:{data:jobs}}));
 await page.locator('.rail [data-view="jobs"]').tap();
 await page.waitForFunction(()=>Number(document.querySelector("#jobRows")?.dataset.renderedRows)>=40);
 for (let batch=0;batch<4 && Number(await page.locator("#jobRows").getAttribute("data-rendered-rows"))<150;batch++) {
   const before=Number(await page.locator("#jobRows").getAttribute("data-rendered-rows"));
   await page.locator(".job-table-wrap").evaluate(e=>e.scrollTop=e.scrollHeight);
   await page.waitForFunction(n=>Number(document.querySelector("#jobRows").dataset.renderedRows)>n,before);
 }
 assert.equal(await page.locator("#jobRows tr:not(.ux-lazy-sentinel)").count(),150);
 const widths=await page.locator(".viewport").evaluate(e=>({width:e.clientWidth,content:e.scrollWidth}));
 assert.ok(widths.content<=widths.width+1);
 results.push("150 job rows and long multilingual wait reasons remain within the mobile viewport");
 await page.route("**/api/workstation/artifacts",r=>r.fulfill({status:503,json:{error:"Synthetic unavailable service <not-html>"}}));
 await page.locator("#mobileMoreButton").tap();await page.locator('#mobileNavSheet [data-view="models"]').tap();
 await page.locator("#toast.show.toast-error").waitFor();
 await page.waitForTimeout(5600);
 assert.ok(await page.locator("#toast.show.toast-error").isVisible());
 assert.match(await page.locator("#toastMessage").innerText(),/<not-html>/);
 assert.equal(await page.locator("#toastMessage not-html").count(),0);
 await page.locator("#toastDismiss").tap();assert.equal(await page.locator("#toast").evaluate(e=>e.classList.contains("show")),false);
 results.push("Service errors remain readable, render as text, and can be dismissed");
 for(const locale of ["zh-Hant","zh-Hans","en"]){
  await page.locator("#studioLocaleButton").tap();await page.locator('#studioLocaleMenu [data-locale="'+locale+'"]').tap();
  await page.locator("#mobileMoreButton").tap();await page.locator('#mobileNavSheet [data-view="settings"]').tap();
  const summary=await page.locator('[data-ux-disclosure="workstationAdvanced"] summary').innerText();
  assert.ok(summary.includes(locale==="en"?"Advanced":locale==="zh-Hant"?"進階":"进阶"));
 }
 results.push("New settings labels switch correctly in English, Traditional Chinese and Simplified Chinese");
 const advanced=page.locator('[data-ux-disclosure="workstationAdvanced"]');
 await advanced.locator("summary").focus();
 const wasOpen=await advanced.evaluate(e=>e.open);
 await page.keyboard.press("Space");
 assert.equal(await page.locator("body").getAttribute("data-current-view"),"settings");
 assert.equal(await advanced.evaluate(e=>e.open),!wasOpen);
 await page.keyboard.press("Enter");
 assert.equal(await advanced.evaluate(e=>e.open),wasOpen);
 results.push("Space and Enter toggle disclosures without triggering numeric navigation shortcuts");

 await page.goto((process.env.STUDIO_QA_BASE || "http://host.docker.internal:9893") + "/synthesis",{waitUntil:"networkidle"});
 assert.equal(await page.locator("body").evaluate(e=>e.classList.contains("embedded")),false);
 assert.equal(await page.locator(".ux-synthesis-details").count(),0);
 results.push("Standalone WebUI does not acquire Studio-only control grouping");
 fs.writeFileSync("/qa/state-results.json",JSON.stringify({passed:results.length,cases:results},null,2));console.log(JSON.stringify(results));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
