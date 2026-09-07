import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { glob } from 'node:fs/promises';

/**
 * SvgTex inlines every MathJax glyph as a full <path>, so a figure-heavy post
 * ships the outline of `x` dozens of times: one post carried 820 paths drawn
 * from 122 distinct glyphs, 343 KB of `d` data.
 *
 * This pass runs over the built HTML and turns every repeated glyph into a
 * <use> pointing at one <defs> copy per page. It is deliberately a post-build
 * step rather than MathJax's own `fontCache`: the cache would have to be
 * scoped to a page, and nothing at render time knows which glyphs a page ends
 * up using until its last component has rendered.
 *
 * Only paths carrying `data-c` are touched — those are MathJax glyphs, which
 * no stylesheet selects and which inherit their fill from the enclosing <g>.
 * Diagram paths keep their element type so that `.edge path` and friends
 * still match.
 */
const GLYPH = /<path data-c="[^"]*" d="([^"]*)"( transform="[^"]*")?><\/path>/g;

export function dedupeMathGlyphs() {
  return {
    name: 'dedupe-math-glyphs',
    hooks: {
      'astro:build:done': async ({ dir, logger }) => {
        let saved = 0;
        for await (const file of glob('**/*.html', { cwd: fileURLToPath(dir) })) {
          const path = fileURLToPath(new URL(file, dir));
          const html = await readFile(path, 'utf8');

          const seen = new Map(); // d -> { count, id }
          for (const m of html.matchAll(GLYPH)) {
            const entry = seen.get(m[1]) ?? { count: 0, id: '' };
            entry.count++;
            seen.set(m[1], entry);
          }
          let n = 0;
          for (const [, entry] of seen) if (entry.count > 1) entry.id = `mj${n++}`;
          if (!n) continue;

          const out = html.replace(GLYPH, (whole, d, transform = '') => {
            const { id } = seen.get(d);
            return id ? `<use href="#${id}"${transform}></use>` : whole;
          });
          const defs = [...seen]
            .filter(([, e]) => e.id)
            .map(([d, e]) => `<path id="${e.id}" d="${d}"></path>`)
            .join('');
          const withDefs = out.replace(
            '</body>',
            `<svg width="0" height="0" aria-hidden="true" style="position:absolute"><defs>${defs}</defs></svg></body>`,
          );
          saved += html.length - withDefs.length;
          await writeFile(path, withDefs);
        }
        logger.info(`deduped MathJax glyphs, ${(saved / 1024).toFixed(0)} KB of HTML saved`);
      },
    },
  };
}
