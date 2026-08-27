// Render the dashboard headlessly and refuse a publish that would drop a
// family on the floor.
//
// The page reads run documents the benches produce. Nothing links the two,
// so a bench can change the shape of what it records and the section that
// draws it silently renders nothing. That is how `rss`, `binary_size` and
// `correctness` came to be collected on every run and shown on none: the
// job stayed green because a blank section is not an error.
//
// This runs the page's own code over the documents about to be published
// and fails when a family that was collected produced no card, so the
// failure lands in CI rather than on the site.
//
// Usage: node tools/perf-site/render_check.mjs <site dir>

import { readFileSync } from "node:fs";

const site = process.argv[2];
if (!site) {
  console.error("usage: render_check.mjs <site dir>");
  process.exit(2);
}

const html = readFileSync(`${site}/index.html`, "utf8");
const script = html.slice(html.indexOf("<script>") + 8, html.lastIndexOf("</script>"));

const nodes = {};
const node = (id) =>
  (nodes[id] ??= { innerHTML: "", value: "", style: {}, addEventListener() {} });

globalThis.document = { querySelector: (sel) => node(sel) };
globalThis.fetch = async (path) => {
  try {
    const body = readFileSync(`${site}/${path}`, "utf8");
    return { ok: true, json: async () => JSON.parse(body) };
  } catch {
    return { ok: false, status: 404 };
  }
};

let index = { runs: [] };
try {
  index = JSON.parse(readFileSync(`${site}/runs/index.json`, "utf8"));
} catch {
  console.error("render check: no runs/index.json, nothing to verify");
  process.exit(1);
}
if (!index.runs?.length) {
  console.error("render check: runs/index.json lists no runs");
  process.exit(1);
}

// Newest against the next newest, which is what the page defaults to.
node("#cur").value = index.runs[0].file;
node("#base").value = index.runs[1]?.file ?? "";
node("#thresh").value = "3";

new Function(script)();
await new Promise((r) => setTimeout(r, 250));
const out = node("#main").innerHTML;

// A family the collector marked anything other than absent has data, so it
// owes the page a section. `absent` is honest and is allowed to draw nothing.
const newest = JSON.parse(readFileSync(`${site}/runs/${index.runs[0].file}`, "utf8"));
const SECTION = {
  throughput: "Throughput",
  isolation: "Transaction isolation",
  correctness: "Transaction correctness",
  rss: "Resident memory",
  binary_size: "Binary size",
  allocs: "Allocations per operation",
};

const failures = [];
for (const [family, heading] of Object.entries(SECTION)) {
  const m = newest.metrics?.[family];
  if (!m || m.trust === "absent") continue;
  if (!out.includes(heading)) {
    failures.push(`${family} was collected (trust=${m.trust ?? "per-entry"}) but "${heading}" is not on the page`);
  }
}
if (!out.includes("Trend across every recorded run")) {
  failures.push("the trend section is missing");
}
if (out.length < 2000) {
  failures.push(`the page rendered only ${out.length} bytes, which is not a populated dashboard`);
}

const cards = (out.match(/<h3>/g) || []).length;
console.log(`render check: ${out.length} bytes, ${cards} cards, ${index.runs.length} run(s) on file`);

if (failures.length) {
  for (const f of failures) console.error(`render check: ${f}`);
  process.exit(1);
}
