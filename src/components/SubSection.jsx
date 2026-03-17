import { useTocRegister } from '../context/TocContext'
import slugify from '../utils/slugify'

export default function SubSection({ title, id, children }) {
  const sectionId = id || slugify(title)
  useTocRegister(sectionId, title, 2)
  return (
    <div className="paper-subsection" id={sectionId}>
      <h3>
        {title}
        <a href={`#${sectionId}`} className="heading-anchor" aria-label="Link to section">#</a>
      </h3>
      {children}
    </div>
  )
}
