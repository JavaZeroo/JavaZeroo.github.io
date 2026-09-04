/**
 * Category names carry a leading emoji (📝学习, 💻Windows …) that is also part
 * of their URL. Keep the slug untouched so Hexo-era links still resolve, but
 * strip the emoji for display.
 */
export const catLabel = (name: string) =>
  name.replace(/^[\p{Extended_Pictographic}\p{Emoji_Presentation}️‍\s]+/u, '') || name;

/** Hand-picked hue per category; anything new falls back to a hash. */
const HUES: Record<string, string> = {
  学习: 'var(--cat-indigo)',
  windows: 'var(--cat-cyan)',
  服务器: 'var(--cat-emerald)',
  python: 'var(--cat-amber)',
};
const FALLBACK = ['var(--cat-indigo)', 'var(--cat-cyan)', 'var(--cat-emerald)', 'var(--cat-amber)', 'var(--cat-rose)'];

export function catColor(name: string): string {
  const key = catLabel(name).toLowerCase();
  if (HUES[key]) return HUES[key];
  let h = 0;
  for (const ch of key) h = (h * 31 + ch.codePointAt(0)!) >>> 0;
  return FALLBACK[h % FALLBACK.length];
}
