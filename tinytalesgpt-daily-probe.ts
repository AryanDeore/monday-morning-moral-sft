// Railway Function: TinyTalesGPT daily probe
// Runs once per cron trigger, checks /api/generate end-to-end latency,
// and sends a Telegram report.

const TARGET_URL =
  process.env.TARGET_URL || "https://tinytalesgpt.aryandeore.ai/api/generate";
const TELEGRAM_BOT_TOKEN = process.env.TELEGRAM_BOT_TOKEN || "";
const TELEGRAM_CHAT_ID = process.env.TELEGRAM_CHAT_ID || "";
const TIMEOUT_MS = Number(process.env.TIMEOUT_MS || 60000);

const payload = {
  topic: "monitoring probe",
  ending: "Happy",
  temperature: 0.7,
};

async function sendTelegram(message: string) {
  if (!TELEGRAM_BOT_TOKEN || !TELEGRAM_CHAT_ID) {
    console.log(
      "Telegram skipped: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing",
    );
    return;
  }

  const url = `https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage`;
  const resp = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      chat_id: TELEGRAM_CHAT_ID,
      text: message,
      parse_mode: "Markdown",
      disable_web_page_preview: true,
    }),
  });

  if (!resp.ok) {
    const txt = await resp.text();
    throw new Error(`Telegram send failed: ${resp.status} ${txt}`);
  }
}

async function probe() {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), TIMEOUT_MS);

  const start = performance.now();
  try {
    const resp = await fetch(TARGET_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    const totalMs = Math.round(performance.now() - start);
    const raw = await resp.text();

    let data: any = null;
    try {
      data = JSON.parse(raw);
    } catch {
      // keep null
    }

    if (!resp.ok) {
      const totalSec = (totalMs / 1000).toFixed(2);
      const msg = [
        `❌ TinyTalesGPT Probe: DOWN`,
        ``,
        `*HTTP:* ${resp.status}`,
        `*E2E latency:* ${totalSec}s`,
        `Body: ${raw.slice(0, 250)}`,
      ].join("\n");
      await sendTelegram(msg);
      return;
    }

    const serverMs =
      typeof data?.latency_ms === "number" ? data.latency_ms : null;
    const totalSec = (totalMs / 1000).toFixed(2);
    const serverSec = serverMs !== null ? (serverMs / 1000).toFixed(2) : "n/a";

    const msg = [
      `✅ TinyTalesGPT Probe: OK`,
      ``,
      `*HTTP:* ${resp.status}`,
      `*E2E latency:* ${totalSec}s`,
      `*Server latency:* ${serverSec}s`,
    ].join("\n");

    await sendTelegram(msg);
  } catch (err: any) {
    const totalMs = Math.round(performance.now() - start);
    const totalSec = (totalMs / 1000).toFixed(2);
    const msg = [
      `❌ TinyTalesGPT Probe: ERROR`,
      ``,
      `*E2E latency before failure:* ${totalSec}s`,
      `Error: ${err?.name || "Error"} - ${err?.message || String(err)}`,
    ].join("\n");
    await sendTelegram(msg);
  } finally {
    clearTimeout(timeout);
  }
}

await probe();
