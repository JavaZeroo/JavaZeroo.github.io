export const SITE_TITLE = 'Java不加糖的Blog';
export const SITE_DESCRIPTION = '学习笔记、项目经验，以及一些折腾记录。';
export const SITE_URL = 'https://blog.javazero.top';
export const AUTHOR = 'Java不加糖';
export const AVATAR = '/images/avatar.jpg';
/** 64px WebP for the 28px header slot; the full avatar stays for the about page. */
export const AVATAR_SMALL = '/images/avatar-64.webp';
export const HOME_POSTS = 7;

/** Hero copy. `tags` render as chips linking at their tag page; `em` is the
 *  emphasised clause of the tagline. */
export const INTRO = {
  tags: [
    { label: '大模型', href: '/tags/大模型/' },
    { label: '模型结构', href: '/tags/模型结构/' },
    { label: '训练系统', href: '/tags/训练系统/' },
    { label: '折腾', href: '/tags/折腾/' },
  ],
  lines: ['读论文和代码，', '把大模型拆成看得懂的图。'],
  em: '偶尔写点数学。',
};

export const SOCIALS = [
  { name: 'GitHub', href: 'https://github.com/JavaZeroo', icon: 'github' },
  { name: '哔哩哔哩', href: 'https://space.bilibili.com/95945127', icon: 'bilibili' },
  { name: 'Telegram', href: 'https://t.me/javaisme0', icon: 'telegram' },
  { name: 'RSS', href: '/atom.xml', icon: 'rss' },
] as const;
