// @ts-check
import mdx from '@astrojs/mdx';
import sitemap from '@astrojs/sitemap';
import pagefind from 'astro-pagefind';
import { unified } from '@astrojs/markdown-remark';
import { defineConfig } from 'astro/config';
import rehypeKatex from 'rehype-katex';
import { rehypeHeadingMath } from './src/plugins/rehype-heading-math.mjs';
import { rehypeMermaidTheme } from './src/plugins/rehype-mermaid-theme.mjs';
import rehypeMermaid from 'rehype-mermaid';
import remarkDirective from 'remark-directive';
import remarkMath from 'remark-math';
import { remarkNote } from './src/plugins/remark-note.mjs';
import { shikiCodeBlock } from './src/plugins/shiki-code-block.mjs';
import { SITE_URL } from './src/consts.ts';

export default defineConfig({
  site: SITE_URL,
  trailingSlash: 'always',
  integrations: [mdx(), sitemap(), pagefind()],
  markdown: {
    // Astro 7 defaults to the Sätteri processor, whose plugins are visitor
    // objects. KaTeX and Mermaid only ship unified transformers, so this site
    // stays on the remark/rehype processor.
    processor: unified({
      // remark-directive must run before remarkNote, which consumes its nodes.
      remarkPlugins: [remarkMath, remarkDirective, remarkNote],
      rehypePlugins: [
        rehypeKatex,
        // Must follow rehypeKatex and precede Astro's heading collection.
        rehypeHeadingMath,
        // Diagrams render to SVG at build time: no client JS, and no pinned CDN
        // that can rot the way the old MathJax one did.
        [rehypeMermaid, { strategy: 'img-svg', mermaidConfig: { theme: 'neutral' }, dark: { theme: 'dark' } }],
        rehypeMermaidTheme,
      ],
      // Off on purpose: this is technical writing full of bare `--flags`, which
      // SmartyPants would rewrite into en dashes.
      smartypants: false,
    }),
    // Shiki would otherwise highlight ```mermaid as a code block before
    // rehype-mermaid ever sees it, leaving a syntax-coloured diagram source.
    syntaxHighlight: { type: 'shiki', excludeLangs: ['mermaid'] },
    shikiConfig: {
      themes: { light: 'catppuccin-latte', dark: 'tokyo-night' },
      defaultColor: false,
      wrap: false,
      transformers: [shikiCodeBlock()],
    },
  },
});
