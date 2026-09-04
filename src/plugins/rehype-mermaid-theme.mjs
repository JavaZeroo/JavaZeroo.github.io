import { visit } from 'unist-util-visit';

const el = (tagName, properties = {}, children = []) => ({ type: 'element', tagName, properties, children });

/**
 * rehype-mermaid's `dark` option emits
 *   <picture><source media="(prefers-color-scheme: dark)" srcset=DARK><img src=LIGHT></picture>
 * which follows the OS setting, not this site's own theme toggle. Rewrite it
 * to two plain images that CSS shows or hides on [data-theme].
 */
export function rehypeMermaidTheme() {
  return (tree) => {
    visit(tree, 'element', (node, index, parent) => {
      if (node.tagName !== 'picture' || index === null || !parent) return;
      const source = node.children.find((c) => c.type === 'element' && c.tagName === 'source');
      const img = node.children.find((c) => c.type === 'element' && c.tagName === 'img');
      if (!source || !img) return;
      const dark = source.properties?.srcSet ?? source.properties?.srcset;
      const light = img.properties?.src;
      if (typeof dark !== 'string' || typeof light !== 'string') return;
      const base = { alt: img.properties?.alt ?? '', loading: 'lazy', decoding: 'async' };
      parent.children[index] = el('figure', { className: ['mermaid'] }, [
        el('img', { ...base, src: light, className: ['mermaid-light'] }),
        el('img', { ...base, src: dark, className: ['mermaid-dark'] }),
      ]);
    });
  };
}
