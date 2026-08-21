// Minimal CDP driver using node's built-in WebSocket (EventTarget API).
const CDP_WS = process.argv[2] || "ws://127.0.0.1:9222/devtools/browser";

class CDP {
  constructor(ws) { this.ws = ws; this.id = 1; this.pending = new Map(); this.handlers = []; }
  send(method, params = {}) {
    const id = this.id++;
    this.ws.send(JSON.stringify({ id, method, params }));
    return new Promise((res, rej) => this.pending.set(id, { res, rej }));
  }
  _onmsg(data) {
    const msg = JSON.parse(data);
    if (msg.id && this.pending.has(msg.id)) {
      const { res, rej } = this.pending.get(msg.id);
      this.pending.delete(msg.id);
      if (msg.error) rej(new Error(JSON.stringify(msg.error))); else res(msg.result);
    } else if (msg.method) {
      for (const h of this.handlers) if (!h.method || h.method === msg.method) h.h(msg);
    }
  }
  on(method, h) { this.handlers.push({ method, h }); }
}

function open(ws) {
  return new Promise((res, rej) => {
    ws.addEventListener("open", () => res(ws));
    ws.addEventListener("error", rej);
  });
}

async function findPage() {
  const cdp = new CDP(await open(new WebSocket(CDP_WS)));
  let pageWsUrl = null;
  for (let i = 0; i < 8; i++) {
    const targets = await cdp.send("Target.listTargets");
    for (const t of targets.targets) {
      if (t.type === "page" && (t.url || "").startsWith("http")) {
        pageWsUrl = t.webSocketDebuggerUrl; break;
      }
    }
    if (pageWsUrl) break;
    await new Promise(r => setTimeout(r, 400));
  }
  return { cdp, pageWsUrl };
}

async function main() {
  const action = process.argv[3];
  const { cdp, pageWsUrl } = await findPage();
  if (!pageWsUrl) { console.error("no page target"); process.exit(1); }
  const page = new CDP(await open(new WebSocket(pageWsUrl)));
  await Promise.all([page.send("Page.enable"), page.send("Runtime.enable"), page.send("Network.enable")]);
  await page.send("Emulation.setDeviceMetricsOverride", { width: 1400, height: 950, deviceScaleFactor: 1 });

  if (action === "nav") {
    await page.send("Page.navigate", { url: process.argv[4] });
    await new Promise(r => setTimeout(r, 4500));
  }
  if (action === "screenshot") {
    const { data } = await page.send("Page.captureScreenshot", { format: "png" });
    const fs = await import("fs"); fs.writeFileSync(process.argv[4], Buffer.from(data, "base64"));
    console.log("wrote", process.argv[4]);
  }
  if (action === "eval") {
    const { result } = await page.send("Runtime.evaluate", {
      expression: process.argv[4], awaitPromise: true, returnByValue: true,
    });
    console.log(JSON.stringify(result?.value));
  }
  if (action === "click") {
    const { result } = await page.send("Runtime.evaluate", {
      expression: `(() => { const el = document.querySelector(${JSON.stringify(process.argv[4])}); if(!el) return 'NOD'; el.scrollIntoView({block:'center'}); el.click(); return 'clicked'; })()`,
      awaitPromise: true, returnByValue: true,
    });
    console.log(JSON.stringify(result?.value));
  }
  if (action === "setinput") {
    // set input value + dispatch input event (for reflex on_change)
    const { result } = await page.send("Runtime.evaluate", {
      expression: `(() => { const el = document.querySelector(${JSON.stringify(process.argv[4])}); const proto = el.tagName==='TEXTAREA'?HTMLTextAreaElement:HTMLInputElement; const setter = Object.getOwnPropertyDescriptor(proto.prototype,'value').set; setter.call(el, ${JSON.stringify(process.argv[5])}); el.dispatchEvent(new Event('input', {bubbles:true})); return 'set'; })()`,
      awaitPromise: true, returnByValue: true,
    });
    console.log(JSON.stringify(result?.value));
  }
  if (action === "wait") {
    await new Promise(r => setTimeout(r, parseInt(process.argv[4] || 3000)));
  }
  if (action === "console") {
    const handler = (msg) => {
      if (msg.method === "Page.console") { const m = msg.params.message; console.log(`[${m.type}] ${m.text}`); }
      if (msg.method === "Page.uncaughtError") { console.log("[UNCAUGHT] " + msg.params.message); }
      if (msg.method === "Runtime.consoleAPICalled" && msg.params.type === "error") {
        console.log("[CONSOLE.ERROR]", (msg.params.args||[]).map(a=>a.value!==undefined?a.value:JSON.stringify(a)).join(", "));
      }
    };
    page.on("Page.console", handler);
    page.on("Page.uncaughtError", handler);
    page.on("Runtime.consoleAPICalled", handler);
    await new Promise(() => {});
  }
  await new Promise(() => {});
}
main().catch(e => { console.error(e); process.exit(1); });
