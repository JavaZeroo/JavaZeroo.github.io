// 截 StepPlayer 的每一帧：桌面浅色、深色各一份，手机 390px 宽每一帧一份。
// 用法：node shot_steps.mjs <文章 slug，如 deepseek-v41-02-csa2> <data-stepplayer 的 id> <输出目录> [端口=4877]
// 先在 dist/ 起 http.server（run_in_background），本机只有 Chromium。
import { chromium } from '/home/ljb/JavaZeroo.github.io/node_modules/playwright/index.mjs';
import { readdirSync, existsSync } from 'node:fs';
const [slug, id, out, port = '4877'] = process.argv.slice(2);
if (!slug || !id || !out) { console.error('用法：node shot_steps.mjs <slug> <player id> <out dir> [port]'); process.exit(1); }
// 文章 URL 从 dist 里找：/YYYY/MM/DD/<slug>/
const root = '/home/ljb/JavaZeroo.github.io/dist';
const find = (dir, depth) => { if (depth === 0) return existsSync(`${dir}/${slug}/index.html`) ? `${dir}/${slug}` : null;
  for (const d of readdirSync(dir, { withFileTypes: true })) if (d.isDirectory() && /^\d+$/.test(d.name)) { const r = find(`${dir}/${d.name}`, depth - 1); if (r) return r; } return null; };
const path = find(root, 3); if (!path) { console.error('dist 里没有这篇文章'); process.exit(1); }
const url = `http://localhost:${port}${path.slice(root.length)}/`;
const b = await chromium.launch();
const shoot = async (ctx, tag, frames, clipMobile) => {
  const p = await ctx.newPage();
  await p.goto(url); await p.waitForTimeout(500);
  const sp = p.locator(`[data-stepplayer="${id}"]`);
  const n = await sp.locator('[data-goto]').count();
  for (const i of frames ?? [...Array(n).keys()]) {
    await sp.locator(`[data-goto="${i}"]`).click(); await p.waitForTimeout(700);
    if (clipMobile) {
      const r = await sp.evaluate((e) => { const r = e.getBoundingClientRect(); return { y: r.top + scrollY, h: r.height, w: innerWidth }; });
      await p.screenshot({ path: `${out}/${id}-${tag}-${i}.png`, fullPage: true, clip: { x: 0, y: r.y, width: r.w, height: Math.min(r.h, 1600) } });
    } else await sp.screenshot({ path: `${out}/${id}-${tag}-${i}.png` });
  }
  console.log(tag, '帧数', n);
};
for (const scheme of ['light', 'dark']) {
  const ctx = await b.newContext({ colorScheme: scheme, viewport: { width: 1280, height: 900 }, deviceScaleFactor: 1.5 });
  await ctx.addInitScript((s) => { try { localStorage.setItem('theme', s); } catch {} }, scheme);
  await shoot(ctx, scheme, null, false);
}
await shoot(await b.newContext({ viewport: { width: 390, height: 844 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true }), 'mobile', null, true);
await b.close();
