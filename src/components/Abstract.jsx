import { useTocRegister } from '../context/TocContext'
import './Abstract.css'

const SECTION_DISPLAY = "TL;DR"
// const SECTION_DISPLAY = "Abstract"
export default function Abstract({ id = 'abstract', children }) {
  useTocRegister(id, SECTION_DISPLAY, 0)
  return (
    <div className="abstract-container" id={id}>
      <h2 className="abstract-title">{SECTION_DISPLAY}</h2>
      <div className="abstract-body">{children}</div>
    </div>
  )
}
