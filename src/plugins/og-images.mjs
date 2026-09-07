import { chromium } from 'playwright';
import sharp from 'sharp';
import { mkdir, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { parse as parseYaml } from 'yaml';

const POSTS = 'src/content/posts';
const SERIES = 'src/content/series';

/** Front matter only; the body is irrelevant to the card. */
async function frontmatter(path) {
  const raw = await readFile(path, 'utf8');
  const m = /^---\r?\n([\s\S]*?)\r?\n---/.exec(raw);
  return m ? parseYaml(m[1]) : {};
}

/**
 * One share image per post, rendered in the same Chromium that already builds
 * the Mermaid diagrams — no extra dependency, and the card gets the site's own
 * typeface by loading the sliced HarmonyOS stylesheet straight out of dist.
 *
 * The files land in dist/og/<post id>.png and are never committed; BaseHead
 * points at that path, which is why the id (the source filename) is the key on
 * both sides.
 */
export function ogImages() {
  return {
    name: 'og-images',
    hooks: {
      'astro:build:done': async ({ dir, logger }) => {
        const root = fileURLToPath(dir);
        const seriesTitle = {};
        for (const f of await readdir(SERIES)) {
          seriesTitle[f.replace(/\.md$/, '')] = (await frontmatter(`${SERIES}/${f}`)).title ?? '';
        }

        const posts = [];
        for (const f of await readdir(POSTS)) {
          if (!/\.mdx?$/.test(f)) continue;
          const fm = await frontmatter(`${POSTS}/${f}`);
          if (fm.draft) continue;
          const kicker = fm.series
            ? [seriesTitle[fm.series], fm.part].filter(Boolean).join(' · ')
            : (fm.categories?.[0] ?? '').replace(/^[\p{Extended_Pictographic}\p{Emoji_Presentation}️‍\s]+/u, '');
          posts.push({ id: f.replace(/\.mdx?$/, ''), title: fm.title ?? '', kicker, date: fmtDate(fm.pubDate) });
        }

        // Written into dist so the card's relative font URLs resolve, and so
        // Chromium is loading everything over file:// from one directory.
        const card = `${root}_og-card.html`;
        await writeFile(card, TEMPLATE);
        await mkdir(`${root}og`, { recursive: true });

        const browser = await chromium.launch();
        const page = await browser.newPage({ viewport: { width: 1200, height: 630 } });
        await page.goto(`file://${card}`);
        await page.evaluate(() => document.fonts.ready);

        for (const post of posts) {
          await page.evaluate((p) => window.render(p), post);
          const shot = await page.screenshot({ type: 'png' });
          await sharp(shot).png({ compressionLevel: 9, palette: true }).toFile(`${root}og/${post.id}.png`);
        }
        await browser.close();
        await rm(card);
        logger.info(`rendered ${posts.length} share images`);
      },
    },
  };
}

const fmtDate = (d) => {
  const t = d instanceof Date ? d : new Date(d);
  return Number.isNaN(+t) ? '' : `${t.getFullYear()}.${String(t.getMonth() + 1).padStart(2, '0')}.${String(t.getDate()).padStart(2, '0')}`;
};

const TEMPLATE = `<!doctype html><meta charset="utf-8">
<link rel="stylesheet" href="fonts/harmonyos-sans-sc/400/result.css">
<link rel="stylesheet" href="fonts/harmonyos-sans-sc/700/result.css">
<style>
  *{box-sizing:border-box}
  body{margin:0;width:1200px;height:630px;overflow:hidden;position:relative;
       background:#0b0f19;color:#e7eaf3;
       font-family:'HarmonyOS Sans SC',system-ui,sans-serif;-webkit-font-smoothing:antialiased}
  .a{position:absolute;border-radius:50%;filter:blur(90px);opacity:.55}
  .a1{width:560px;height:560px;left:-140px;top:-200px;background:#6366f1}
  .a2{width:460px;height:460px;right:-120px;top:-60px;background:#22d3ee}
  .a3{width:420px;height:420px;left:560px;bottom:-260px;background:#8b5cf6}
  .grid{position:absolute;inset:0;background-image:radial-gradient(rgba(255,255,255,.13) 1px,transparent 1px);
        background-size:28px 28px;-webkit-mask-image:linear-gradient(#000,transparent 72%);mask-image:linear-gradient(#000,transparent 72%)}
  .frame{position:absolute;inset:0;padding:78px 96px 150px;display:flex;flex-direction:column;justify-content:center}
  .bar{width:64px;height:5px;border-radius:3px;background:linear-gradient(90deg,#818cf8,#22d3ee)}
  .kicker{margin-top:26px;font-size:26px;font-weight:500;letter-spacing:.02em;color:#22d3ee;
          white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  h1{margin:22px 0 0;font-weight:700;letter-spacing:-.02em;line-height:1.24;color:#e7eaf3;
     display:-webkit-box;-webkit-box-orient:vertical;-webkit-line-clamp:3;overflow:hidden}
  .foot{position:absolute;left:96px;right:96px;bottom:72px;display:flex;align-items:center;gap:16px;font-size:24px;color:#8b93a8}
  .foot img{width:44px;height:44px;border-radius:50%;border:1px solid rgba(255,255,255,.18)}
  .foot b{color:#aab2c5;font-weight:500}
  .foot .sp{flex:1}
</style>
<body>
  <div class="a a1"></div><div class="a a2"></div><div class="a a3"></div><div class="grid"></div>
  <div class="frame">
    <div class="bar"></div>
    <div class="kicker" id="k"></div>
    <h1 id="t"></h1>
  </div>
  <div class="foot">
      <img src="images/avatar-64.webp" alt="">
      <b>Java不加糖</b><span>blog.javazero.top</span>
    <span class="sp"></span><span id="d"></span>
  </div>
<script>
  const t = document.getElementById('t');
  window.render = ({ title, kicker, date }) => {
    document.getElementById('k').textContent = kicker || '';
    document.getElementById('d').textContent = date || '';
    t.textContent = title;
    // Longer titles step down until three lines fit the space above the footer.
    for (const size of [76, 68, 60, 54, 48, 42]) {
      t.style.fontSize = size + 'px';
      if (t.scrollHeight <= t.clientHeight + 1) break;
    }
  };
</script>
`;
