// One-off: derive favicons, a small header avatar, and the default share image
// from public/images/avatar.jpg. Outputs are committed; rerun after changing
// the avatar or the site name.
//   node scripts/gen-assets.mjs
import sharp from 'sharp';
import { chromium } from 'playwright';
import { writeFileSync, mkdirSync } from 'node:fs';

const SRC = 'public/images/avatar.jpg';
mkdirSync('public/icons', { recursive: true });

for (const [size, name] of [[32, 'favicon-32.png'], [180, 'apple-touch-icon.png']]) {
  await sharp(SRC).resize(size, size).png({ compressionLevel: 9 }).toFile(`public/icons/${name}`);
}
// Header/hero avatar at 2x of its 28px slot.
await sharp(SRC).resize(64, 64).webp({ quality: 82 }).toFile('public/images/avatar-64.webp');
console.log('icons + small avatar written');

// Share card: rendered from HTML so it uses the site's own type and gradient.
const html = `<!doctype html><meta charset="utf-8"><style>
  @font-face{font-family:H;src:url(file://${process.cwd()}/node_modules/.cache/hos-bold.ttf)}
  body{margin:0;width:1200px;height:630px;background:#0b0f19;color:#e7eaf3;font-family:'HarmonyOS Sans SC',system-ui,sans-serif;position:relative;overflow:hidden}
  .a{position:absolute;border-radius:50%;filter:blur(90px);opacity:.6}
  .a1{width:620px;height:620px;left:-120px;top:-180px;background:#6366f1}.a2{width:520px;height:520px;right:-80px;top:40px;background:#22d3ee}.a3{width:400px;height:400px;left:520px;bottom:-220px;background:#8b5cf6}
  .grid{position:absolute;inset:0;background-image:radial-gradient(rgba(255,255,255,.14) 1px,transparent 1px);background-size:28px 28px;mask-image:linear-gradient(#000,transparent 80%)}
  .c{position:absolute;left:96px;top:150px}
  .k{font:500 22px/1 monospace;letter-spacing:.1em;color:#22d3ee;margin-bottom:34px}
  h1{margin:0 0 26px;font-size:112px;line-height:1;letter-spacing:-.03em;font-weight:700;background:linear-gradient(120deg,#818cf8,#22d3ee);-webkit-background-clip:text;color:transparent}
  p{margin:0;font-size:34px;color:#aab2c5}
  .u{position:absolute;left:96px;bottom:70px;font-size:24px;color:#6c7590;font-family:monospace}
</style><body><div class="a a1"></div><div class="a a2"></div><div class="a a3"></div><div class="grid"></div>
<div class="c"><div class="k">深度学习 · 点云 · 折腾记录</div><h1>Java不加糖</h1><p>记录深度学习、点云，以及各种折腾。偶尔写点数学。</p></div>
<div class="u">blog.javazero.top</div></body>`;
const br = await chromium.launch(); const pg = await br.newPage({ viewport: { width: 1200, height: 630 } });
await pg.setContent(html); await pg.waitForTimeout(300);
const png = await pg.screenshot({ type: 'png' }); await br.close();
writeFileSync('public/og.png', await sharp(png).png({ compressionLevel: 9, palette: true }).toBuffer());
console.log('og.png written');
