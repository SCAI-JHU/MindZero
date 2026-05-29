import { useEffect } from 'react'
import { NumberingProvider } from './context/NumberingContext'
import { TocProvider, useTocSections } from './context/TocContext'
import Header from './components/Header'
import Sidebar from './components/Sidebar'
import Content from './content'
import { paperMeta } from './paperMeta'

function numberSections(sections) {
  let sectionNum = 0
  let subNum = 0
  return sections.map((s) => {
    if (s.level === 0) {
      return { ...s }
    }
    if (s.level === 1) {
      sectionNum++
      subNum = 0
      return { ...s, title: `${sectionNum}. ${s.title}` }
    }
    if (s.level === 2) {
      subNum++
      return { ...s, title: `${sectionNum}.${subNum} ${s.title}` }
    }
    return { ...s }
  })
}

function AppContent() {
  const tocSections = useTocSections()
  const numberedSections = numberSections(tocSections)

  useEffect(() => {
    document.title = paperMeta.title
  }, [])

  return (
    <div className="layout">
      <Sidebar sections={numberedSections} />

      <main className="main-content">
        <Header
          title={paperMeta.title}
          venue={paperMeta.venue}
          authors={paperMeta.authors}
          affiliations={paperMeta.affiliations}
          links={paperMeta.links}
        />

        <Content />

        <footer className="footer">
          <p>
            MindZero · <a href="https://scai.cs.jhu.edu/" target="_blank" rel="noopener noreferrer">SCAI Lab</a> · Johns Hopkins University
          </p>
        </footer>
      </main>
    </div>
  )
}

export default function App() {
  return (
    <NumberingProvider>
      <TocProvider>
        <AppContent />
      </TocProvider>
    </NumberingProvider>
  )
}
