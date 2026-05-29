import Abstract from './components/Abstract'
import Section from './components/Section'
import SubSection from './components/SubSection'
import Figure from './components/Figure'
import Table from './components/Table'
import CodeBlock from './components/CodeBlock'
import Math from './components/Math'
import citationCode from './assets/code/citation.bib?raw'
import assistanceExampleImg from './assets/figures/assistance-example.svg'
import methodSsrlImg from './assets/figures/method-ssrl.svg'
import methodRewardImg from './assets/figures/method-reward.svg'
import experimentsImg from './assets/figures/experiments.png'
import gwQaImg from './assets/figures/gw-qa-main.svg'
import householdQaImg from './assets/figures/embodied-qa-main.svg'
import progressGwImg from './assets/figures/progress-gw.svg'
import progressHouseholdImg from './assets/figures/progress-embodied.svg'
import humanInterfaceImg from './assets/figures/human-experiment-interface.png'
import './content.css'

function ContributionGrid() {
  const contributions = [
    {
      title: 'No Annotations',
      body: 'Learns from behavior without ground-truth goals or beliefs.',
    },
    {
      title: 'Efficient Inference',
      body: 'Runs online mental reasoning in a single pass after training.',
    },
    {
      title: 'Multiple Domains',
      body: 'Evaluated in both GridWorld and household environments.',
    },
    {
      title: 'Assistance Gains',
      body: '24.5% GridWorld speedup vs. 0.0% for Gemini-3-Flash.',
    },
  ]

  return (
    <div className="contribution-grid" aria-label="MindZero main contributions">
      {contributions.map((item) => (
        <div className="contribution-item" key={item.title}>
          <div className="contribution-title">{item.title}</div>
          <p>{item.body}</p>
        </div>
      ))}
    </div>
  )
}

function FigureRow({ panels, caption }) {
  return (
    <figure className="figure-container figure-row">
      <div className="figure-row-panels">
        {panels.map((panel) => (
          <div className="figure-row-panel" key={panel.title}>
            <img src={panel.src} alt={panel.alt || panel.title} />
            {panel.title && <div className="figure-row-title">{panel.title}</div>}
          </div>
        ))}
      </div>
      <figcaption className="figure-caption">{caption}</figcaption>
    </figure>
  )
}

export default function Content() {
  return (
    <>
      <Figure
        label="fig:teaser"
        src={assistanceExampleImg}
        caption="Online mental reasoning for proactive household assistance. As actions arrive, MindZero updates probabilities over multiple goal hypotheses and helps when confidence supports action."
        width="100%"
      />

      <ContributionGrid />

      <Abstract>
        <p>
          MindZero is a self-supervised reinforcement learning framework that trains multimodal language models to perform efficient online Theory-of-Mind reasoning. Instead of requiring ground-truth annotations of hidden goals or beliefs, MindZero learns from behavior: it proposes mental-state hypotheses, receives reward when those hypotheses explain observed actions, and internalizes model-based reasoning into fast single-pass inference after training.
        </p>
      </Abstract>

      <Section title="Motivation">
        <p>
          Real-world assistants need to infer what people want from what they do. This is harder than answering a single Theory-of-Mind question: an assistant must continuously update uncertainty over competing hypotheses, act early enough to be useful, and avoid committing to an incorrect goal too soon.
        </p>
        <p>
          Model-based Theory-of-Mind methods, such as Bayesian inverse planning, are robust and interpretable but too costly for online assistance. Learning-based methods can be fast, but reliable labels for hidden mental states are difficult to collect in realistic domains. MindZero targets this gap: it learns the structure of model-based reasoning directly from behavior, without mental-state annotations.
        </p>
        <p>
          We formalize online mental-state inference as Bayesian inference over the latest mental state <Math>{String.raw`m_t`}</Math> given states <Math>{String.raw`s_{1:t}`}</Math> and actions <Math>{String.raw`a_{1:t}`}</Math>:
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
        <SubSection title="Self-Supervised RL">
          <p>
            MindZero frames mental reasoning as self-supervised reinforcement learning centered on explanatory consistency. Instead of predicting the next action directly, the model generates mental-state hypotheses that should explain the actions already observed.
          </p>
          <FigureRow
            panels={[
              { title: 'Self-Supervised RL', src: methodSsrlImg },
              { title: 'Reward Computation', src: methodRewardImg },
            ]}
            caption="MindZero's training loop and reward computation. Model outputs are evaluated by action likelihoods, mental priors, and an entropy term before GRPO updates."
          />
        </SubSection>

        <SubSection title="Reward Design">
          <p>
            Given a behavior history, the model outputs a finite set of mental-state hypotheses and normalized probabilities. Action likelihood rewards hypotheses that make observed behavior probable, mental priors discourage implausible goals, and entropy keeps the posterior diverse under ambiguity.
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
          <p>
            This gives the model a self-supervised signal for learning explicit, uncertainty-aware mental reasoning without requiring labels for goals, beliefs, or intentions.
          </p>
        </SubSection>
      </Section>

      <Section title="Experiments">
        <p>
          We evaluate MindZero across four settings spanning direct mental-state question answering and embodied proactive assistance in GridWorld and household environments.
        </p>
        <Figure
          label="fig:experiments"
          src={experimentsImg}
          caption="Experimental settings: GridWorld question answering, GridWorld proactive assistance, household question answering, and household proactive assistance."
          width="100%"
        />

        <SubSection title="Question Answering">
          <p>
            In both domains, MindZero improves small open-weight backbones while keeping inference cost close to single-pass model use. It reaches 95.0% and 92.3% accuracy in GridWorld QA with Qwen3-VL-4B and Qwen3-VL-8B, and up to 77.8% in household QA.
          </p>
          <FigureRow
            panels={[
              { title: 'GridWorld Question Answering', src: gwQaImg },
              { title: 'Household Question Answering', src: householdQaImg },
            ]}
            caption="Question answering accuracy and inference cost. MindZero improves base models by 1.7-2.5x and outperforms test-time scaling baselines in the reported settings."
          />
        </SubSection>

        <SubSection title="Proactive Assistance">
          <p>
            In proactive assistance, the helper must infer goals at every step and act under uncertainty. MindZero turns online mental reasoning into useful help: in GridWorld, MindZero with Qwen3-VL-8B reaches 24.5% speedup, while Gemini-3-Flash yields 0.0% speedup because its goal predictions are unstable across time.
          </p>
          <div className="table-pair">
            <Table
              label="tab:gridworld-assist"
              caption="GridWorld proactive assistance."
              headers={['Method', 'Speedup ↑', 'TFLOPs ↓']}
              rows={[
                ['Random Goal', '0.0', 'N/A'],
                ['Qwen3-VL-4B', '1.4', '151.7'],
                ['Qwen3-VL-8B', '-0.1', '295.2'],
                ['Qwen3-VL-235B-A22B', '1.0', '808.6'],
                ['GPT-5.2', '0.0', 'Proprietary'],
                ['Gemini-3-Flash', '0.0', 'Proprietary'],
                ['MindZero w/ Qwen3-VL-4B', '23.0', '161.4'],
                ['MindZero w/ Qwen3-VL-8B', <strong>24.5</strong>, '291.8'],
              ]}
            />
            <Table
              label="tab:household-assist"
              caption="Household proactive assistance."
              headers={['Method', 'Speedup ↑', 'TFLOPs ↓']}
              rows={[
                ['Random Goal', '-2.2', 'N/A'],
                ['Llama-3.2-3B*', '2.3', '244.3'],
                ['Llama-3.1-8B', '1.7', '656.1'],
                ['Qwen3-4B', '2.3', '213.1'],
                ['Qwen3-235B-A22B', '12.3', '1101.6'],
                ['GPT-5.2', '9.4', 'Proprietary'],
                ['Gemini-3-Flash', '17.7', 'Proprietary'],
                ['MindZero w/ Llama-3.2-3B*', '4.3', '235.1'],
                ['MindZero w/ Llama-3.1-8B', '17.4', '608.4'],
                ['MindZero w/ Qwen3-4B', <strong>19.1</strong>, <strong>201.2</strong>],
              ]}
            />
          </div>
          <p className="table-note">
            * indicates models that require format warm-up before RL training because they cannot reliably generate goal hypotheses in the required format.
          </p>
        </SubSection>

        <SubSection title="Online Goal Inference">
          <p>
            MindZero's predictions become more accurate as task progress reveals more evidence, while most baselines stay low or improve too late to support useful assistance.
          </p>
          <FigureRow
            panels={[
              { title: 'GridWorld Proactive Assistance', src: progressGwImg },
              { title: 'Household Proactive Assistance', src: progressHouseholdImg },
            ]}
            caption="Goal accuracy and F1 over task progress. MindZero maintains stable online improvement in both GridWorld and household assistance."
          />
        </SubSection>
      </Section>

      <Section title="Ablation">
        <p>
          The household ablation isolates three design choices: explicit prior modeling, multiple hypotheses, and entropy regularization. Removing any of them reduces proactive assistance speedup, showing that MindZero needs both plausible hypotheses and calibrated uncertainty.
        </p>
        <Table
          label="tab:ablation"
          caption="Ablation on household proactive assistance using Qwen3-4B."
          headers={['#', 'Method', 'Speedup ↑', 'TFLOPs ↓']}
          rows={[
            ['I', <strong>MindZero</strong>, <strong>19.1</strong>, '201.2'],
            ['II', 'w/o prior modeling', '17.0', '200.5'],
            ['III', 'w/o multiple hypotheses', '10.3', <strong>132.6</strong>],
            ['IV', 'w/o entropy bonus', '5.2', '245.1'],
          ]}
        />
      </Section>

      <Section title="Human Study">
        <p>
          We also evaluate MindZero with real users in the household assistance domain. Twelve Johns Hopkins University participants completed simulated household tasks under four settings: single human, assistance with Qwen3-4B, assistance with MindZero trained from Qwen3-4B, and assistance with Gemini-3-Flash.
        </p>
        <Figure
          label="fig:human-study"
          src={humanInterfaceImg}
          caption="Human experiment interface for household proactive assistance."
          width="100%"
        />
        <p>
          MindZero trained from Qwen3-4B achieves 19.7% average speedup over the single-human baseline, compared with 2.6% for the pretrained Qwen3-4B backbone. Gemini-3-Flash reaches 23.4%, and the difference between Gemini-3-Flash and MindZero is not statistically significant under a paired t-test on speedup.
        </p>
      </Section>

      <Section title="Citation" numbered={false}>
        <CodeBlock filename="citation.bib">{citationCode}</CodeBlock>
      </Section>
    </>
  )
}
