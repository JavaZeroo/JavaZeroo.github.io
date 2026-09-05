import katex from 'katex';

/** 构建期把一段 LaTeX 渲染成 KaTeX 的 HTML。 */
export const tex = (src: string, display = false) =>
  katex.renderToString(src, { throwOnError: false, output: 'html', displayMode: display, strict: 'ignore' });

const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

/** 中文与公式混排：`$...$` 之间的部分走 KaTeX，其余原样输出。 */
export const texMix = (src: string) =>
  src.split(/(\$[^$]+\$)/g).map((seg) => (seg.startsWith('$') && seg.endsWith('$') ? tex(seg.slice(1, -1)) : esc(seg))).join('');
