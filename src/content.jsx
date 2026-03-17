import Abstract from './components/Abstract'
import Section from './components/Section'
import SubSection from './components/SubSection'
import Figure from './components/Figure'
import Table from './components/Table'
import Video from './components/Video'
import CodeBlock from './components/CodeBlock'
import Math from './components/Math'
import Ref from './components/Ref'
import citationCode from './assets/code/citation.bib?raw'

export const paperMeta = {
  title: 'MindZero: Learning Online Mental Reasoning With Zero Annotations',
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
  links: [
    { label: 'Paper', icon: 'ai ai-arxiv', url: 'https://arxiv.org' },
    { label: 'Code', icon: 'fa-brands fa-github', url: 'https://github.com/SCAI-JHU/MindZero' },
    { label: 'Data', icon: 'fa-brands fa-hugging-face', url: 'https://huggingface.co/datasets/SCAI-JHU/MindZero' },
    { label: 'Models', icon: 'fa-brands fa-hugging-face', url: 'https://huggingface.co/SCAI-JHU' },
  ],
}


export default function Content() {
  return (
    <>
      <Abstract>
        <p>
          MindZero is a self-supervised reinforcement learning framework that trains multimodal large language models (MLLMs) for efficient and robust online mental reasoning.
        </p>
      </Abstract>

      <Section title="Introduction">
        <p>
          Effective real-world assistance requires AI agents with robust Theory of Mind (ToM): inferring human mental states from their behavior.
        </p>
        <p>
          we formalize online mental state inference as following Bayesian inference:
        </p>
        <Math display>
          {String.raw`
\underbrace{P(m_t \mid s_{1:t}, a_{1:t})}_\text{posterior}
\propto
\underbrace{P(a_{1:t} \mid m_t, s_{1:t})}_\text{action likelihood}
\cdot
\underbrace{P(m_t)}_\text{prior}.
`}
        </Math>
      </Section>

      <Section title="Method">
        <SubSection title="Self-Supervised RL for Mental Reasoning">
          <p>
            we formulate mental reasoning as a self-supervised reinforcement learning (SSRL) problem centered on explanatory consistency.
            Instead of treating actions as prediction targets, we view them as evidence, as shown in <Ref label="fig:method" />.
          </p>
          <Figure
            label="fig:method"
            src="https://raw.githubusercontent.com/SCAI-JHU/MindZero/main/assets/framework.png"
            caption="Overview"
            width="100%"
          />
        </SubSection>

        <SubSection title="Reward Design">
          <p>
            The optimization objective can be formalized as the following reward function:
          </p>
          <Math display>
            {String.raw`
\mathcal{J}(\theta)=\mathbb{E}_{Q_\theta}
[ \log (
P(a_{1:t} \mid m_t, s_{1:t})
\cdot
P(m_t)
)]
+ H(Q_\theta).
`}
          </Math>
        </SubSection>
      </Section>

      <Section title="Experiments">
        <p>
          To understand the key components driving MindZero's performance, we conduct comprehensive ablation studies on Qwen3-4B, as shown in <Ref label="tab:ablation" />.
        </p>
        <Table
          label="tab:ablation"
          caption="Ablation"
          headers={['#', 'Method', 'Speedup ↑', 'TFLOPs ↓']}
          rows={[
            ['I', 'MindZero', '26.3', '201.2'],
            ['II', 'w/o prior modeling', '17.0', '200.5'],
            ['III', 'w/o multiple hypotheses', '10.3', '132.6'],
            ['IV', 'w/o entropy bonus', '5.2', '245.1'],
          ]}
        />
      </Section >

      <Section title="Citation" numbered={false}>
        <CodeBlock filename="citation.bib">{citationCode}</CodeBlock>
      </Section>
    </>
  )
}
