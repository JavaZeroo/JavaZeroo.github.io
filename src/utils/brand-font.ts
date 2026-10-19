import { readFileSync } from 'node:fs';
import { AUTHOR } from '../consts';

// Build-time only: retain the existing licensed font shards, but declare just
// the characters in the brand. Article text uses the reader's system fonts.
const sheet = readFileSync('public/fonts/harmonyos-sans-sc/700/result.css', 'utf8');
const points = [...new Set([...AUTHOR].map((char) => char.codePointAt(0)!))];
export const brandFontCss = [...sheet.matchAll(/@font-face\{([^}]+)\}/g)].flatMap(([, rule]) => {
  const file = /url\("\.\/([^"/]+\.woff2)"\)/.exec(rule)?.[1];
  const ranges = /unicode-range:([^;]+)/.exec(rule)?.[1].split(',').map((range) => {
    const [start, end] = range.trim().replace(/^U\+/i, '').split('-').map((hex) => parseInt(hex, 16));
    return [start, end ?? start];
  });
  if (!file || !ranges) return [];
  const chars = points.filter((point) => ranges.some(([start, end]) => point >= start && point <= end));
  if (!chars.length) return [];
  return `@font-face{font-family:'Blog Brand';font-style:normal;font-weight:700;font-display:optional;src:url('/fonts/harmonyos-sans-sc/700/${file}') format('woff2');unicode-range:${chars.map((point) => `U+${point.toString(16)}`).join(',')}}`;
}).join('\n');
