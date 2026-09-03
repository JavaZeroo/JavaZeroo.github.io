import { visit } from 'unist-util-visit';

const HEADINGS = new Set(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']);

const hasClass = (node, name) => {
  const raw = node.properties?.className ?? [];
  return (Array.isArray(raw) ? raw : [raw]).includes(name);
};

/**
 * Inside headings only, drop KaTeX's MathML twin and unhide its visual output.
 *
 * KaTeX emits the same formula three times over: MathML, the original LaTeX in
 * an <annotation>, and the visual HTML. Astro harvests heading text after this
 * plugin runs, so without it a heading like `### $y'=f(x)$` reaches the table
 * of contents as "y′=f(x)y'=f(x)y′=f(x)". Keeping the visual copy and removing
 * `aria-hidden` leaves one readable copy for both the TOC and screen readers.
 */
export function rehypeHeadingMath() {
  return (tree) => {
    visit(tree, 'element', (node) => {
      if (!HEADINGS.has(node.tagName)) return;
      visit(node, 'element', (inner) => {
        if (!hasClass(inner, 'katex')) return;
        inner.children = inner.children.filter((c) => !hasClass(c, 'katex-mathml'));
        for (const child of inner.children) {
          if (hasClass(child, 'katex-html')) delete child.properties['ariaHidden'];
        }
      });
      return 'skip';
    });
  };
}
