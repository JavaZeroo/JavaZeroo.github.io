import assert from 'node:assert/strict';
import { cp, mkdir, mkdtemp, readFile, readdir, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import test from 'node:test';
import { ogImages } from '../src/plugins/og-images.mjs';

test('share image cache reuses unchanged cards and invalidates visual inputs', async () => {
  const original = process.cwd();
  const root = await mkdtemp(join(tmpdir(), 'blog-og-test-'));
  try {
    await mkdir(join(root, 'src/content/posts'), { recursive: true });
    await mkdir(join(root, 'src/content/series'), { recursive: true });
    await cp(join(original, 'package-lock.json'), join(root, 'package-lock.json'));
    process.chdir(root);
    const post = 'src/content/posts/example.md';
    const source = (title, body = 'Body', draft = false) => `---\ntitle: ${title}\npubDate: 2026-01-01\nseries: sample\ndraft: ${draft}\n---\n${body}\n`;
    await writeFile('src/content/series/sample.md', '---\ntitle: Series\n---\n');
    const build = async (changeFont = false) => {
      await rm('dist', { recursive: true, force: true });
      await mkdir('dist/images', { recursive: true });
      await cp(join(original, 'public/fonts/harmonyos-sans-sc'), 'dist/fonts/harmonyos-sans-sc', { recursive: true });
      await cp(join(original, 'public/images/avatar-64.webp'), 'dist/images/avatar-64.webp');
      if (changeFont) {
        const css = 'dist/fonts/harmonyos-sans-sc/700/result.css';
        await writeFile(css, `${await readFile(css, 'utf8')}\n/* updated font stylesheet */\n`);
      }
      const messages = [];
      await ogImages().hooks['astro:build:done']({
        dir: pathToFileURL(`${root}/dist/`),
        logger: { info: (message) => messages.push(message) },
      });
      assert.ok(!(await readdir('dist')).includes('_og-card.html'));
      return messages.at(-1);
    };
    await writeFile(post, source('First title'));
    assert.equal(await build(), 'reused 0 share images; rendered 1');
    const first = await readFile('dist/og/example.png');
    assert.equal(await build(), 'reused 1 share images; rendered 0');
    assert.deepEqual(await readFile('dist/og/example.png'), first);
    await writeFile(post, source('First title', 'Changed body'));
    assert.equal(await build(), 'reused 1 share images; rendered 0');
    await writeFile(post, source('Changed title'));
    assert.equal(await build(), 'reused 0 share images; rendered 1');
    assert.notDeepEqual(await readFile('dist/og/example.png'), first);
    await writeFile('src/content/series/sample.md', '---\ntitle: Changed series\n---\n');
    assert.equal(await build(), 'reused 0 share images; rendered 1');
    // A dependency change must invalidate cards even when their metadata matches.
    await writeFile('package-lock.json', '{}');
    assert.equal(await build(), 'reused 0 share images; rendered 1');
    assert.equal(await build(true), 'reused 0 share images; rendered 1');
    assert.equal(await build(true), 'reused 1 share images; rendered 0');
    await writeFile(post, source('Changed title', 'Body', true));
    assert.equal(await build(), 'reused 0 share images; rendered 0');
    assert.deepEqual(await readdir('dist/og'), []);
  } finally {
    process.chdir(original);
    await rm(root, { recursive: true, force: true });
  }
});
