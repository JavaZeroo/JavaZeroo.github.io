import { getCollection, type CollectionEntry } from 'astro:content';

export type Post = CollectionEntry<'posts'>;

/**
 * Hexo permalink shape: /:year/:month/:day/:title/ where :title was the
 * filename.
 *
 * Read the date with UTC getters, always. YAML parses an unquoted timestamp
 * with no offset as UTC, so UTC getters give back exactly the calendar date
 * that was typed, on any machine. Local getters would shift every post written
 * after 16:00 to the next day on a UTC+8 box and not on a UTC CI runner --
 * i.e. permalinks that differ between laptop and CI.
 */
export function permalinkOf(post: Post): string {
  const d = post.data.pubDate;
  const yyyy = d.getUTCFullYear();
  const mm = String(d.getUTCMonth() + 1).padStart(2, '0');
  const dd = String(d.getUTCDate()).padStart(2, '0');
  return `${yyyy}/${mm}/${dd}/${post.id}`;
}

export const urlOf = (post: Post) => `/${permalinkOf(post)}/`;

/**
 * Drafts are visible while running `astro dev` so they can be previewed, and
 * dropped from every production build.
 */
const isVisible = ({ data }: Post) => import.meta.env.DEV || !data.draft;

/** Visible posts, newest first, sticky ones pinned to the top. */
export async function getPosts(): Promise<Post[]> {
  const posts = await getCollection('posts', isVisible);
  return posts.sort((a, b) => {
    if (a.data.sticky !== b.data.sticky) return b.data.sticky - a.data.sticky;
    return b.data.pubDate.valueOf() - a.data.pubDate.valueOf();
  });
}

/** Strictly chronological, ignoring sticky. Used by archives and feeds. */
export async function getPostsByDate(): Promise<Post[]> {
  const posts = await getCollection('posts', isVisible);
  return posts.sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf());
}

export function countBy(posts: Post[], key: 'categories' | 'tags'): Map<string, Post[]> {
  const map = new Map<string, Post[]>();
  for (const post of posts) {
    for (const value of post.data[key]) {
      const bucket = map.get(value);
      if (bucket) bucket.push(post);
      else map.set(value, [post]);
    }
  }
  return new Map([...map].sort((a, b) => b[1].length - a[1].length));
}

export function groupByYear(posts: Post[]): [string, Post[]][] {
  const map = new Map<string, Post[]>();
  for (const post of posts) {
    const year = String(post.data.pubDate.getUTCFullYear());
    const bucket = map.get(year);
    if (bucket) bucket.push(post);
    else map.set(year, [post]);
  }
  return [...map].sort((a, b) => Number(b[0]) - Number(a[0]));
}

export const describe = (post: Post) => post.data.description || post.data.title;
