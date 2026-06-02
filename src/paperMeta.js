export const paperMeta = {
  title: 'MindZero: Learning Online Mental Reasoning With Zero Annotations',
  venue: 'ICML 2026',
  authors: [
    { name: 'Shunchi Zhang', affiliations: ['1'], equal: true, url: 'https://shunchi.dev' },
    { name: 'Jin Lu', affiliations: ['1'], equal: true, url: 'https://www.linkedin.com/in/ryanlu19/' },
    { name: 'Chuanyang Jin', affiliations: ['1'], equal: true, url: 'https://chuanyangjin.com' },
    { name: 'Yichao Zhou', affiliations: ['2'], equal: true, url: 'https://github.com/ZYClbt' },
    { name: 'Zhining Zhang', affiliations: ['2'], url: 'https://zhining.space' },
    { name: 'Tianmin Shu', affiliations: ['1'], url: 'https://tshu.io' },
  ],
  affiliations: [
    'Johns Hopkins University',
    'Peking University',
  ],
  // Each inner array is one row: first row = core artifacts, second row = outreach.
  links: [
    [
      { label: 'Paper', icon: 'fa-solid fa-file', url: 'https://arxiv.org/pdf/2606.00240' },
      { label: 'Code', icon: 'fa-brands fa-github', url: 'https://github.com/SCAI-JHU/MindZero' },
      { label: 'Data & Models', icon: 'fa-solid fa-database', url: 'https://huggingface.co/collections/SCAI-JHU/mindzero' },
    ],
    [
      { label: 'Slides', icon: 'fa-solid fa-chalkboard', url: './slides/' },
      { label: 'Talk', icon: 'fa-solid fa-video', url: 'https://recorder-v3.slideslive.com/?share=112361&s=9f8eaaf6-e910-44f2-98fb-2d1852389bfb' },
    ],
  ],
}
