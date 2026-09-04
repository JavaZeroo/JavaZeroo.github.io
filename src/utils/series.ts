import { getCollection, type CollectionEntry } from 'astro:content';
import type { Post } from './posts';

export type Series = CollectionEntry<'series'>;

export const seriesUrl = (id: string) => `/series/${id}/`;

export const getAllSeries = () => getCollection('series');

/** Every post in a series, drafts included, in reading order. */
export async function seriesPosts(series: Series): Promise<Post[]> {
  const posts = await getCollection('posts', ({ data }) => data.series === series.id);
  const rank = new Map(series.data.parts.map((p, i) => [p, i]));
  for (const post of posts) {
    // Fail loudly at build time: a typo here would silently drop the post
    // from its chapter.
    if (!post.data.part || !rank.has(post.data.part)) {
      throw new Error(
        `Post "${post.id}" names part "${post.data.part}" but series "${series.id}" only defines: ${series.data.parts.join(', ')}`,
      );
    }
  }
  return posts.sort((a, b) => {
    const pa = rank.get(a.data.part!)!, pb = rank.get(b.data.part!)!;
    if (pa !== pb) return pa - pb;
    if (a.data.order !== b.data.order) return a.data.order - b.data.order;
    return a.data.pubDate.valueOf() - b.data.pubDate.valueOf();
  });
}

/** Reading order grouped by part, keeping empty parts so the plan is visible. */
export async function seriesOutline(series: Series): Promise<[string, Post[]][]> {
  const posts = await seriesPosts(series);
  return series.data.parts.map((part) => [part, posts.filter((p) => p.data.part === part)]);
}

/** Where a post sits in its series, counting published posts only. */
export async function seriesContext(post: Post) {
  if (!post.data.series) return null;
  const series = (await getAllSeries()).find((s) => s.id === post.data.series);
  if (!series) throw new Error(`Post "${post.id}" references unknown series "${post.data.series}"`);
  const published = (await seriesPosts(series)).filter((p) => !p.data.draft || import.meta.env.DEV);
  const i = published.findIndex((p) => p.id === post.id);
  return {
    series,
    index: i + 1,
    total: published.length,
    prev: i > 0 ? published[i - 1] : undefined,
    next: i >= 0 && i < published.length - 1 ? published[i + 1] : undefined,
  };
}
