
const { chromium } = require("playwright");
const fs = require("fs");
(async () => {
 const browser = await chromium.launch({headless:true,args:["--disable-gpu"]});
 const results=[]; fs.mkdirSync("/qa/shots",{recursive:true});
 for (const [name,width,height] of [["desktop",1440,900],["laptop",1280,720],["phone",390,844],["small-phone",360,800],["landscape",844,390],["tablet",768,1024]]) {
  const context=await browser.newContext({viewport:{width,height},hasTouch:width<=860,isMobile:width<=860,locale:"en-US",reducedMotion:"reduce"});
  const page=await context.newPage(); const errors=[];page.on("pageerror",e=>errors.push(e.message));
  await page.goto("http://host.docker.internal:9893/",{waitUntil:"networkidle"});
  await page.locator("body.studio-ux").waitFor();
  for(const view of ["overview","synthesis","expressions","datasets","tse","training","evaluation","models","engines","jobs","settings"]){
   await page.evaluate(v=>{location.hash=v},view);
   await page.locator('[data-view-panel="'+view+'"].active').waitFor();
   await page.waitForTimeout(180);
   const dimensions=await page.evaluate(()=>{
    const v=document.querySelector(".viewport");
    return {body:document.documentElement.scrollWidth,screen:innerWidth,viewport:v.clientWidth,content:v.scrollWidth,
     activeTitle:document.querySelector(".view.active h1")?.textContent,
     controls:[...document.querySelectorAll(".view.active button")].filter(x=>x.getBoundingClientRect().width>0).length};
   });
   if(["desktop","phone","landscape"].includes(name)) await page.screenshot({path:`/qa/shots/${name}-${view}.png`});
   results.push({name,view,...dimensions,errors:[...errors]});
  }
  await context.close();
 }
 await browser.close();fs.writeFileSync("/qa/layout-results.json",JSON.stringify(results,null,2));
 console.log(JSON.stringify(results.filter(x=>x.content>x.viewport+1||x.body>x.screen||x.errors.length),null,2));
 console.log("Checked",results.length,"module/viewports");
})().catch(e=>{console.error(e);process.exit(1)});
