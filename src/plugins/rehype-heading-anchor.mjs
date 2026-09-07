import Slugger from 'github-slugger';
import { visit } from 'unist-util-visit';

const rawNodeTypes = new Set(['text', 'raw', 'mdxTextExpression']);
const codeTagNames = new Set(['code', 'pre']);

/**
 * Gives every body heading a permalink handle.
 *
 * The id has to be assigned here rather than left to Astro's own
 * `rehypeHeadingIds`, because the anchor needs to know the slug to link to.
 * Astro keeps an id that is already set, and the text extraction below is a
 * copy of its own, so the slugs — and therefore every existing deep link —
 * stay exactly what they were.
 *
 * The anchor carries no text: Astro harvests heading text for the table of
 * contents by walking every descendant text node, so a literal "#" inside the
 * heading would leak into the TOC. The glyph comes from CSS instead.
 */
export function rehypeHeadingAnchor() {
  return (tree, file) => {
    const slugger = new Slugger();
    const isMDX = /\.mdx$/.test(file.history[0] ?? '');
    visit(tree, 'element', (node) => {
      if (!/^h[1-6]$/.test(node.tagName)) return;
      let text = '';
      visit(node, (child, __, parent) => {
        if (child.type === 'element' || parent == null) return;
        if (child.type === 'raw' && /^\n?<.*>\n?$/.test(child.value)) return;
        if (rawNodeTypes.has(child.type) && 'value' in child) {
          const inCode = 'tagName' in parent && codeTagNames.has(parent.tagName);
          text += isMDX || inCode ? child.value : child.value.replace(/\{/g, '${');
        }
      });
      node.properties ??= {};
      if (typeof node.properties.id !== 'string') node.properties.id = slugger.slug(text);
      node.children.push({
        type: 'element',
        tagName: 'a',
        properties: { className: ['h-anchor'], href: `#${node.properties.id}`, 'aria-label': `链接到本节：${text}` },
        children: [],
      });
      return 'skip';
    });
  };
}
