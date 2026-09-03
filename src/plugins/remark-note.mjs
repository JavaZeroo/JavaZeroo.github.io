import { visit } from 'unist-util-visit';

const TYPES = new Set(['info', 'success', 'warning', 'danger']);

/**
 * `:::note{type="warning"}` container directives -> <aside class="note note-warning">.
 * Content inside stays Markdown, so bold/lists/math still render.
 */
export function remarkNote() {
  return (tree) => {
    visit(tree, (node) => {
      if (node.type !== 'containerDirective' || node.name !== 'note') return;
      const raw = node.attributes?.type ?? 'info';
      const type = TYPES.has(raw) ? raw : 'info';
      const data = (node.data ||= {});
      data.hName = 'aside';
      data.hProperties = { className: ['note', `note-${type}`] };
    });
  };
}
