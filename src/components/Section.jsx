import { useTocRegister } from '../context/TocContext'
import slugify from '../utils/slugify'
import './Section.css'

export default function Section({ title, id, numbered = true, children }) {
  const sectionId = id || slugify(title)
  useTocRegister(sectionId, title, numbered ? 1 : 0)
  return (
    <section className={`paper-section${numbered ? '' : ' paper-section-unnumbered'}`} id={sectionId}>
      <h2>
        {title}
        <a href={`#${sectionId}`} className="heading-anchor" aria-label="Link to section">#</a>
      </h2>
      {children}
    </section>
  )
}
