// One-off: slice HarmonyOS Sans SC into unicode-range woff2 chunks.
//
// Not wired into the build on purpose. cn-font-split pulls a platform binary
// from GitHub on install, which is exactly the kind of moving part CI should
// not depend on; the output is committed instead and only regenerated when the
// typeface or the weight set changes.
//
//   npx -p cn-font-split@7 node scripts/split-fonts.mjs <dir-with-HarmonyOS_SansSC_*.ttf>
import { readFileSync, rmSync, mkdirSync } from 'node:fs';
import { join } from 'node:path';
import { fontSplit } from 'cn-font-split';

const src = process.argv[2];
if (!src) throw new Error('usage: split-fonts.mjs <dir containing HarmonyOS_SansSC_*.ttf>');

const WEIGHTS = { Regular: 400, Medium: 500, Bold: 700 };
const OUT = 'public/fonts/harmonyos-sans-sc';

for (const [style, weight] of Object.entries(WEIGHTS)) {
  const outDir = join(OUT, String(weight));
  rmSync(outDir, { recursive: true, force: true });
  mkdirSync(outDir, { recursive: true });
  await fontSplit({
    input: new Uint8Array(readFileSync(join(src, `HarmonyOS_SansSC_${style}.ttf`))),
    outDir,
    // ~120 KB per slice: roughly half the slice count of the default, so the
    // @font-face stylesheet stays small; a page still fetches only a few.
    chunkSize: 120 * 1024,
    css: { fontFamily: 'HarmonyOS Sans SC', fontWeight: String(weight), fontDisplay: 'swap', compress: true },
    renameOutputFont: '[hash:8].[ext]',
    testHtml: false,
    reporter: false,
    previewImage: false,
    silent: true,
  });
  console.log(`weight ${weight} -> ${outDir}`);
}
