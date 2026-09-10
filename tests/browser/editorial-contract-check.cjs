
const {chromium}=require('playwright');const fs=require('fs');const assert=require('node:assert/strict');
(async()=>{
const browser=await chromium.launch({headless:true});const page=await browser.newPage({viewport:{width:1440,height:900},locale:'en-US'});const response=await page.goto((process.env.STUDIO_QA_BASE || 'http://host.docker.internal:9893')+'/#training',{waitUntil:'networkidle'});assert.equal(response.headers()['x-aniflive-ui-fixture'],'synthetic');
const rail=page.locator('.rail');const geometry=()=>page.evaluate(()=>({w:document.querySelector('.rail').getBoundingClientRect().width,x:document.querySelector('.viewport').getBoundingClientRect().left,icon:document.querySelector('.dock-home svg').getBoundingClientRect().x}));
await page.mouse.move(500,100);await page.waitForTimeout(250);const compact=await geometry();assert.equal(compact.w,72);
await rail.hover({position:{x:35,y:80}});await page.waitForTimeout(50);const mid=await geometry();await page.waitForTimeout(220);const expanded=await geometry();assert.ok(mid.w>72&&mid.w<172);assert.equal(Math.round(expanded.w),172);assert.equal(expanded.x,compact.x);assert.ok(Math.abs(expanded.icon-compact.icon)<1);
const clipped=await page.locator('.rail .nav-label').evaluateAll(es=>es.filter(e=>e.getBoundingClientRect().right>e.parentElement.getBoundingClientRect().right).map(e=>e.textContent));assert.deepEqual(clipped,[]);
await page.mouse.move(500,100);await page.waitForTimeout(50);const closing=await geometry();assert.ok(closing.w>72&&closing.w<172);
await rail.hover({position:{x:35,y:80}});await page.waitForTimeout(30);await page.mouse.move(500,100);await page.waitForTimeout(240);assert.equal((await geometry()).w,72);
await page.locator('.dock-home').focus();await page.keyboard.press('Tab');await page.waitForTimeout(240);assert.equal(Math.round((await geometry()).w),172);
await page.emulateMedia({reducedMotion:'reduce'});assert.equal(await rail.evaluate(e=>getComputedStyle(e).transitionDuration),'0s');
const nav={compact,mid,expanded,closing,labelsFit:true,rapidReversal:true,keyboard:true,reducedMotion:true};
await page.mouse.move(500,100);await page.locator('.view.active h1').click();
const previous=await page.locator('#trainingRows').innerText();
await page.route('**/api/workstation/projects',r=>r.fulfill({status:503,json:{error:'QA read failure'}}));
await page.locator('.rail [data-view="jobs"]').click();await page.locator('.rail [data-view="training"]').click();
await page.locator('.view.active .ux-data-error').first().waitFor();assert.equal(await page.locator('#trainingRows').innerText(),previous);assert.match(await page.locator('.view.active .ux-data-error').first().innerText(),/previously loaded/);
await page.unroute('**/api/workstation/projects');
const initialFailure=await browser.newPage({viewport:{width:1440,height:900},locale:'en-US'});
await initialFailure.route('**/api/workstation/projects',r=>r.fulfill({status:503,json:{error:'QA first read failure'}}));
await initialFailure.goto((process.env.STUDIO_QA_BASE || 'http://host.docker.internal:9893')+'/#training',{waitUntil:'networkidle'});
assert.match(await initialFailure.locator('#trainingCount').innerText(),/Unavailable/);
await initialFailure.unroute('**/api/workstation/projects');
await initialFailure.route('**/api/workstation/jobs',async r=>{const response=await r.fetch();const payload=await response.json();payload.data=payload.data.map(j=>({...j,progress:null}));await r.fulfill({json:payload})});
await initialFailure.route('**/api/workstation/jobs/*',async r=>{const response=await r.fetch();const payload=await response.json();if(payload.job)payload.job.progress=null;await r.fulfill({json:payload})});
await initialFailure.reload({waitUntil:'networkidle'});
const progressCells=await initialFailure.locator('#trainingRows tr td:nth-child(4)').allTextContents();
assert.ok(progressCells.length>0&&progressCells.every(v=>v==='Unavailable'),JSON.stringify(progressCells));
await initialFailure.close();
const result={nav,readFailureRetainsRows:true,firstReadFailureDoesNotShowZero:true,unknownProgressDoesNotShowZero:true,reflow:[]};
for(const zoom of [100,125,150,200]){
 await page.setViewportSize({width:Math.floor(1440*100/zoom),height:Math.floor(900*100/zoom)});
 for(const view of ['overview','synthesis','expressions','datasets','tse','training','evaluation','models','engines','jobs','settings']){
  await page.evaluate(v=>location.hash=v,view);await page.locator('[data-view-panel="'+view+'"].active').waitFor();await page.waitForTimeout(60);
  const d=await page.evaluate(()=>{const e=document.querySelector('.viewport');return {w:e.clientWidth,content:e.scrollWidth,body:document.documentElement.scrollWidth,screen:innerWidth}});
  assert.ok(d.content<=d.w+1&&d.body<=d.screen+1,JSON.stringify({zoom,view,...d}));result.reflow.push({equivalentZoom:zoom,view,...d});
 }
}
result.reflowMethod='CSS viewport equivalent to 1440x900 at zoom; not native browser zoom certification';
fs.writeFileSync('/qa/editorial-extended-results.json',JSON.stringify(result,null,2));console.log(JSON.stringify({nav,readFailureRetainsRows:true,reflowCases:result.reflow.length}));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});
