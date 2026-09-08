import { visit } from 'unist-util-visit';

const TYPES = new Set(['info', 'success', 'warning', 'danger']);

/**
 * `:::note{type="warning"}` container directives -> <aside class="note note-warning">.
 * `:::details[标题]` container directives -> <details class="fold"><summary>标题</summary>…</details>,
 * collapsed by default (add `{open}` to start expanded).
 * Content inside stays Markdown, so bold/lists/math still render.
 */
export function remarkNote() {
  return (tree) => {
    visit(tree, (node) => {
      if (node.type !== 'containerDirective') return;
      const data = (node.data ||= {});
      if (node.name === 'note') {
        const raw = node.attributes?.type ?? 'info';
        const type = TYPES.has(raw) ? raw : 'info';
        data.hName = 'aside';
        data.hProperties = { className: ['note', `note-${type}`] };
        return;
      }
      if (node.name === 'details') {
        data.hName = 'details';
        data.hProperties = { className: ['fold'], open: node.attributes?.open != null ? true : undefined };
        const first = node.children[0];
        if (first?.type === 'paragraph' && first.data?.directiveLabel) {
          first.data.hName = 'summary';
        } else {
          node.children.unshift({
            type: 'paragraph',
            data: { hName: 'summary' },
            children: [{ type: 'text', value: node.attributes?.summary ?? '展开' }],
          });
        }
      }
    });
  };
}
