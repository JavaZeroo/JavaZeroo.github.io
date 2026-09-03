import rss from '@astrojs/rss';
import { getPostsByDate, urlOf, describe } from './posts';
import { SITE_TITLE, SITE_DESCRIPTION } from '../consts';

/**
 * Both /atom.xml and /rss.xml serve the same RSS 2.0 document. The Hexo site
 * published its feed at /atom.xml, so that path has to keep working for
 * existing subscribers; feed readers dispatch on the root element, not the
 * filename, so serving RSS there is safe.
 */
export async function buildFeed(context: { site?: URL | undefined }) {
  const posts = await getPostsByDate();
  return rss({
    title: SITE_TITLE,
    description: SITE_DESCRIPTION,
    site: context.site!,
    items: posts.map((post) => ({
      title: post.data.title,
      pubDate: post.data.pubDate,
      description: describe(post),
      link: urlOf(post),
      categories: [...post.data.categories, ...post.data.tags],
    })),
  });
}
