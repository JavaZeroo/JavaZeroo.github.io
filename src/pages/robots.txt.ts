import type { APIContext } from 'astro';

/** Generated rather than dropped in public/, so the sitemap URL follows SITE_URL. */
export const GET = ({ site }: APIContext) =>
  new Response(`User-agent: *\nAllow: /\n\nSitemap: ${new URL('sitemap-index.xml', site)}\n`, {
    headers: { 'Content-Type': 'text/plain; charset=utf-8' },
  });
