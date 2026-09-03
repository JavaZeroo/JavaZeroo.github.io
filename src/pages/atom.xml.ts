import type { APIContext } from 'astro';
import { buildFeed } from '../utils/feed';

export const GET = (context: APIContext) => buildFeed(context);
