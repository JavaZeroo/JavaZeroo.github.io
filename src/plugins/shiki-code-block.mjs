const el = (tagName, properties = {}, children = []) => ({
  type: 'element',
  tagName,
  properties,
  children,
});

const copyIcon = () =>
  el(
    'svg',
    {
      viewBox: '0 0 24 24',
      fill: 'none',
      stroke: 'currentColor',
      strokeWidth: 2,
      strokeLinecap: 'round',
      strokeLinejoin: 'round',
      'aria-hidden': 'true',
    },
    [
      el('rect', { x: 9, y: 9, width: 12, height: 12, rx: 2 }),
      el('path', { d: 'M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1' }),
    ],
  );

/**
 * Wraps every highlighted block in a chrome carrying its language and a copy
 * button.
 *
 * This has to be a Shiki transformer, not a rehype plugin: Astro highlights
 * code during the remark pass and splices the result in as raw HTML, so by the
 * time rehype plugins run there is no `<pre>` element left to match on.
 */
export function shikiCodeBlock() {
  return {
    name: 'code-block-chrome',
    root(node) {
      const pre = node.children.find((c) => c.type === 'element');
      if (!pre) return;
      const lang = this.options.lang || 'text';

      const head = el('div', { className: ['code-head'] }, [
        el('span', { className: ['code-lang'] }, [{ type: 'text', value: lang }]),
        el(
          'button',
          { type: 'button', className: ['copy-btn'], 'data-copy': '', 'aria-label': '复制代码' },
          [copyIcon(), el('span', { 'data-copy-label': '' }, [{ type: 'text', value: '复制' }])],
        ),
      ]);

      node.children = [el('div', { className: ['code-block'] }, [head, pre])];
    },
  };
}
