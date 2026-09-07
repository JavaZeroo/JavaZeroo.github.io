import katex from 'katex';
import { mathjax } from 'mathjax-full/js/mathjax.js';
import { TeX } from 'mathjax-full/js/input/tex.js';
import { SVG } from 'mathjax-full/js/output/svg.js';
import { liteAdaptor } from 'mathjax-full/js/adaptors/liteAdaptor.js';
import { RegisterHTMLHandler } from 'mathjax-full/js/handlers/html.js';
import { AllPackages } from 'mathjax-full/js/input/tex/AllPackages.js';

// ---------------------------------------------------------------------------
// HTML 场景（figcaption、demo 图例）：KaTeX → HTML。
// ---------------------------------------------------------------------------

/** 构建期把一段 LaTeX 渲染成 KaTeX 的 HTML。 */
export const tex = (src: string, display = false) =>
  katex.renderToString(src, { throwOnError: false, output: 'html', displayMode: display, strict: 'ignore' });

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** 中文与公式混排：`$...$` 之间的部分走 KaTeX，其余原样输出。 */
export const texMix = (src: string) =>
  src.split(/(\$[^$]+\$)/g).map((seg) => (seg.startsWith('$') && seg.endsWith('$') ? tex(seg.slice(1, -1)) : esc(seg))).join('');

// ---------------------------------------------------------------------------
// SVG 场景（图里的标签）：MathJax → 纯 SVG 路径。
// 不用 <foreignObject>：Safari 对缩放过的 foreignObject 里的定位元素会算错坐标，
// KaTeX 的上下标全靠 position: relative，一进 Safari 就整体漂移。
// ---------------------------------------------------------------------------

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);
const mjDoc = mathjax.document('', {
  InputJax: new TeX({ packages: AllPackages }),
  // fontCache: 'none' 把字形路径直接内联，没有 <defs>/<use>，多个公式放进同一个 SVG 不会撞 id。
  OutputJax: new SVG({ fontCache: 'none' }),
});

export interface MathSvg {
  /** <svg> 内部的内容（一个 <g> 加若干 <path>） */
  inner: string;
  /** MathJax 的 viewBox，单位 1000 = 1em，y 轴向上为负（基线在 0） */
  vb: [number, number, number, number];
}

const mathCache = new Map<string, MathSvg>();

/** 把一段 LaTeX 渲染成 SVG 路径。 */
export function mathSvg(src: string): MathSvg {
  const hit = mathCache.get(src);
  if (hit) return hit;
  const node = mjDoc.convert(src, { display: false });
  const out = adaptor.outerHTML(node);
  const m = out.match(/<svg\b[^>]*\bviewBox="([^"]+)"[^>]*>([\s\S]*?)<\/svg>/);
  if (!m) throw new Error(`MathJax 没有产出 SVG：${src}`);
  const vb = m[1].trim().split(/\s+/).map(Number) as [number, number, number, number];
  const res = { inner: m[2], vb };
  mathCache.set(src, res);
  return res;
}

/** 估算一个字符在无衬线字体里的宽度（em）。站点字体随平台变化，这里只能估。 */
function charWidth(ch: string): number {
  const c = ch.codePointAt(0)!;
  if (c === 0x20) return 0.28;
  if (c >= 0x30 && c <= 0x39) return 0.56;                 // 数字
  if (c >= 0x41 && c <= 0x5a) return 0.7;                  // 大写
  if (c >= 0x61 && c <= 0x7a) return 0.56;                 // 小写
  if ('=+<>~'.includes(ch)) return 0.58;
  if ('()[]{}/\\|!'.includes(ch)) return 0.3;
  if (',.:;\'`'.includes(ch)) return 0.28;
  if (c < 0x80) return 0.4;                                // 其余 ASCII 标点
  if (c === 0xd7 || c === 0xf7 || c === 0xb1) return 0.58;   // × ÷ ±
  if (c === 0xb7) return 0.28;                             // ·
  if ((c >= 0x2e80 && c <= 0x9fff) || (c >= 0xf900 && c <= 0xfaff) || (c >= 0xff00 && c <= 0xffef) || (c >= 0x3000 && c <= 0x303f)) return 1.0; // CJK 与全角标点
  if (c >= 0x2000 && c <= 0x206f) return 0.5;              // 一般标点（… ‰ 等）
  if (c >= 0x2190 && c <= 0x21ff) return 1.0;              // 箭头
  if (c >= 0x2200 && c <= 0x22ff) return 0.6;              // 数学运算符（− ≈ ≤ ∈）
  if (c >= 0x2300 && c <= 0x2bff) return 0.9;              // 其他符号
  return 0.6;
}

export const textWidth = (s: string, size: number, weight = 400) =>
  [...s].reduce((acc, ch) => acc + charWidth(ch), 0) * size * (weight >= 600 ? 1.06 : 1);

export type LabelSeg =
  | { kind: 'text'; text: string; x: number; width: number }
  | { kind: 'math'; inner: string; vb: [number, number, number, number]; x: number; width: number; height: number; ascent: number };

/**
 * 把一条中英混排的标签排成一行：`$...$` 之间走 MathJax，其余按估算宽度排布。
 * 返回每段相对于行首的 x 偏移和整行宽度，供组件做 start / middle / end 对齐。
 */
export function layoutLabel(src: string, size: number, weight = 400, mathScale = 1.1): { segs: LabelSeg[]; width: number } {
  const segs: LabelSeg[] = [];
  let x = 0;
  for (const part of src.split(/(\$[^$]+\$)/g)) {
    if (!part) continue;
    if (part.length > 2 && part.startsWith('$') && part.endsWith('$')) {
      const { inner, vb } = mathSvg(part.slice(1, -1));
      const k = (size * mathScale) / 1000;
      const pad = size * 0.1; // 公式两侧留一点空，MathJax 的包围盒是紧的
      const seg = { kind: 'math' as const, inner, vb, x: x + pad, width: vb[2] * k, height: vb[3] * k, ascent: -vb[1] * k };
      segs.push(seg);
      x += seg.width + 2 * pad;
    } else {
      const width = textWidth(part, size, weight);
      segs.push({ kind: 'text', text: part, x, width });
      x += width;
    }
  }
  return { segs, width: x };
}
