import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { readFile, stat, mkdir } from 'node:fs/promises';
import { resolve, extname, sep } from 'node:path';
import { chromium } from 'playwright';

// Exercise the production output, including Pagefind, without a dev server or
// another test framework. Use an ephemeral port so local previews can coexist.
const root = resolve('dist');
await stat(resolve(root, 'index.html'));
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.wasm': 'application/wasm', '.svg': 'image/svg+xml', '.woff2': 'font/woff2', '.png': 'image/png', '.webp': 'image/webp' };
const server = createServer(async (req, res) => {
  try {
    let file = resolve(root, `.${decodeURIComponent(new URL(req.url, 'http://localhost').pathname)}`);
    if (file !== root && !file.startsWith(root + sep)) { res.writeHead(403).end(); return; }
    if ((await stat(file)).isDirectory()) file = resolve(file, 'index.html');
    res.setHeader('Content-Type', types[extname(file)] ?? 'application/octet-stream');
    res.end(await readFile(file));
  } catch { res.writeHead(404).end(); }
});
await new Promise((done) => server.listen(0, '127.0.0.1', done));
const base = `http://127.0.0.1:${server.address().port}`;
const browser = await chromium.launch();
const article = '/2026/10/02/deepseek-v41-02-csa2/';
const errors = [];
async function context(options = {}) {
  const ctx = await browser.newContext({ reducedMotion: 'reduce', ...options });
  // Third-party comments are not part of this deterministic smoke check.
  await ctx.route('https://giscus.app/**', (route) => route.fulfill({ body: '', contentType: 'text/javascript' }));
  const page = await ctx.newPage();
  page.on('pageerror', (error) => errors.push(error.message));
  page.setDefaultTimeout(10000);
  return { ctx, page };
}
async function screenshot(page, name) {
  if (!process.env.SMOKE_SCREENSHOTS) return;
  await mkdir(process.env.SMOKE_SCREENSHOTS, { recursive: true });
  await page.screenshot({ path: resolve(process.env.SMOKE_SCREENSHOTS, `${name}.png`) });
}
async function noOverflow(page) {
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'page must fit viewport');
}
try {
  for (const width of [1440, 390, 320]) {
    const { ctx, page } = await context({ viewport: { width, height: 900 } });
    await page.goto(base);
    await page.evaluate(() => document.fonts.ready);
    const fonts = await page.evaluate(() => performance.getEntriesByType('resource').filter((r) => r.name.includes('.woff2')).map((r) => ({ url: r.name, bytes: r.decodedBodySize })));
    const bytes = fonts.reduce((sum, font) => sum + font.bytes, 0);
    assert.ok(bytes < 300000, `home font budget exceeded: ${bytes}`);
    assert.ok(!fonts.some((font) => font.url.includes('/400/')), 'body must not download HarmonyOS regular');
    console.log(`Home ${width}px: ${fonts.length} font requests, ${bytes} bytes`);
    await noOverflow(page);
    await screenshot(page, `home-${width}`);
    await page.goto(base + article);
    await page.locator('.prose [data-stepplayer][data-ready]').first().waitFor();
    await noOverflow(page);
    if (width < 640) {
      assert.ok(await page.locator('.series-banner .count').evaluate((el) => el.getBoundingClientRect().height < 30), 'series count stays on one line');
    }
    await screenshot(page, `article-${width}-light`);
    await page.locator('[data-theme-toggle]').click();
    assert.equal(await page.locator('html').getAttribute('data-theme'), 'dark');
    await screenshot(page, `article-${width}-dark`);

    const player = page.locator('.prose [data-stepplayer]:not(.fixed)').first();
    const id = await player.getAttribute('data-stepplayer');
    await player.locator('.fig-zoom').click();
    const dialog = page.getByRole('dialog', { name: '放大图表' });
    assert.equal(await page.locator(`[data-stepplayer="${id}"]`).count(), 1, 'zoom must not clone IDs or controls');
    await dialog.locator('[data-next]').click();
    assert.equal(await dialog.locator('[data-stepplayer]').getAttribute('data-step'), '1');
    for (let i = 0; i < 12; i++) {
      await page.keyboard.press('Tab');
      assert.ok(await dialog.evaluate((el) => el.contains(document.activeElement)), 'focus stays inside modal');
    }
    await page.keyboard.press('Escape');
    assert.equal(await dialog.count(), 0);
    assert.equal(await player.getAttribute('data-step'), '1', 'zoom preserves state on close');
    assert.ok(await player.locator('.fig-zoom').evaluate((el) => el === document.activeElement));
    assert.equal(await page.evaluate(() => document.documentElement.style.overflow), '');
    await player.locator('[data-next]').click();
    assert.equal(await player.getAttribute('data-step'), '2', 'restored figure remains interactive');
    await player.locator('.fig-zoom').click();
    await dialog.locator('.fig-close').click();
    assert.ok(await player.locator('.fig-zoom').evaluate((el) => el === document.activeElement));
    await player.locator('.fig-zoom').click();
    await page.goBack();
    await page.waitForURL(base + '/');
    await page.locator('[data-hero]').waitFor();
    assert.equal(await page.locator('.fig-lightbox').count(), 0, 'navigation closes modal');
    assert.equal(await page.evaluate(() => document.documentElement.style.overflow), '');
    await page.goForward();
    await page.waitForURL(base + article);
    await page.locator('.prose [data-stepplayer][data-ready]').first().waitFor();

    // A real client-side navigation and back must reinitialize all controls.
    await page.locator('.brand').click();
    await page.waitForURL(base + '/');
    await page.locator('[data-hero]').waitFor();
    assert.equal(await page.locator('html').getAttribute('data-theme'), 'dark');
    await page.goBack();
    await page.waitForURL(base + article);
    await page.locator('.prose [data-stepplayer][data-ready]').first().waitFor();
    await page.locator('[data-search-open]').click();
    await page.locator('[data-search-input]').fill('KV');
    await page.locator('[data-search-results] a').first().waitFor();
    await page.keyboard.press('Escape');
    assert.ok(await page.locator('[data-search-open]').evaluate((el) => el === document.activeElement));
    console.log(`PASS ${width}px: layout, themes, figure controls, modal focus, navigation, real search`);
    await ctx.close();
  }

  // Resolve old searches after new ones, including data() and close/reopen.
  const { ctx, page } = await context();
  await ctx.route('**/pagefind/pagefind.js*', (route) => route.fulfill({ contentType: 'text/javascript', body: `
    window.pendingSearches = {};
    window.pendingData = {};
    export async function init() {}
    export function search(q) { if(q === 'failure') return Promise.reject(new Error('network')); return new Promise(resolve => {
      window.pendingSearches[q] = () => resolve({results:[{data: () => new Promise(done => {
        window.pendingData[q] = () => done({url:'/',meta:{title:q},excerpt:q});
      })}]});
    }); }
  ` }));
  await page.goto(base);
  await page.locator('[data-search-open]').click();
  const input = page.locator('[data-search-input]');
  const waitSearch = (q) => page.waitForFunction((q) => !!window.pendingSearches?.[q], q);
  const resolveSearch = (q) => page.evaluate((q) => window.pendingSearches[q](), q);
  const waitData = (q) => page.waitForFunction((q) => !!window.pendingData?.[q], q);
  const resolveData = (q) => page.evaluate((q) => window.pendingData[q](), q);
  await input.fill('old'); await waitSearch('old'); await resolveSearch('old'); await waitData('old');
  await input.fill('new'); await waitSearch('new'); await resolveSearch('new'); await waitData('new'); await resolveData('new');
  await page.waitForFunction(() => document.querySelector('.st')?.textContent === 'new');
  await resolveData('old');
  assert.equal(await page.locator('.st').textContent(), 'new');
  await input.fill('clear'); await waitSearch('clear');
  await input.fill(''); await resolveSearch('clear');
  assert.equal(await page.locator('[data-search-results]').textContent(), '');
  await input.fill('closed'); await waitSearch('closed');
  await page.keyboard.press('Escape');
  await page.locator('[data-search-open]').click();
  await resolveSearch('closed');
  assert.equal(await page.locator('[data-search-results]').textContent(), '');
  await input.fill('failure');
  await page.getByText('搜索暂时不可用，请稍后重新输入重试', { exact: true }).waitFor();
  assert.equal(await page.locator('[data-search-results]').getAttribute('aria-busy'), 'false');
  await ctx.close();
  console.log('PASS search: stale results, clear input, close/reopen, query failure');

  const retry = await context();
  let loads = 0;
  await retry.ctx.route('**/pagefind/pagefind.js*', (route) => {
    loads++;
    return route.fulfill({ contentType: 'text/javascript', body: loads === 1
      ? 'export async function init(){throw new Error("offline")}'
      : 'export async function init(){} export async function search(){return {results:[]}}' });
  });
  await retry.page.goto(base);
  await Promise.all([
    retry.page.waitForEvent('console', (message) => message.text().includes('[search] pagefind failed to load')),
    retry.page.locator('[data-search-open]').click(),
  ]);
  await retry.page.locator('[data-search-input]').fill('retry');
  await retry.page.getByText('没有找到相关文章', { exact: true }).waitFor();
  assert.equal(loads, 2);
  await retry.ctx.close();
  console.log('PASS search: failed initialization can retry');
  assert.deepEqual(errors, [], 'no uncaught browser errors');
} finally {
  await browser.close();
  await new Promise((done) => server.close(done));
}
