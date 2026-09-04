import { visit } from 'unist-util-visit';

/**
 * Small fixes to rendered post bodies:
 * - images load lazily and decode off the main thread;
 * - an `<h1>` inside a post body is demoted to `<h2>`. The page already has
 *   one h1 (the title); several migrated posts open with a Markdown `#`
 *   heading that would otherwise duplicate it.
 */
export function rehypePostPolish() {
  return (tree) => {
    visit(tree, 'element', (node) => {
      if (node.tagName === 'img') {
        node.properties.loading ??= 'lazy';
        node.properties.decoding ??= 'async';
      } else if (node.tagName === 'h1') {
        node.tagName = 'h2';
      }
    });
  };
}
