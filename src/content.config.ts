import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const posts = defineCollection({
  loader: glob({
    base: './src/content/posts',
    pattern: '**/*.{md,mdx}',
    // Keep the filename verbatim as the id. Hexo built permalinks from the
    // filename, so anything else here would break every existing URL.
    generateId: ({ entry }) => entry.replace(/\.(md|mdx)$/, ''),
  }),
  schema: z.object({
    title: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    description: z.string().default(''),
    categories: z.array(z.string()).default([]),
    tags: z.array(z.string()).default([]),
    sticky: z.number().default(0),
    draft: z.boolean().default(false),
    /** Membership in a series: which one, which part, and position within the part. */
    series: z.string().optional(),
    part: z.string().optional(),
    order: z.number().default(0),
  }),
});

/**
 * A series is a curated, ordered reading path over ordinary posts. Posts keep
 * their own URLs; the series only adds a table of contents and in-series
 * prev/next. `parts` fixes the chapter order; a post naming a part that is
 * not listed fails the build (see utils/series.ts).
 */
const series = defineCollection({
  loader: glob({ base: './src/content/series', pattern: '*.md', generateId: ({ entry }) => entry.replace(/\.md$/, '') }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    parts: z.array(z.string()).min(1),
    color: z.string().default('var(--accent)'),
  }),
});

export const collections = { posts, series };
