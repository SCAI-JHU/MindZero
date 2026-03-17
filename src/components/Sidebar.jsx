import { useState } from 'react'
import { useScrollSpy } from '../hooks/useScrollSpy'
import './Sidebar.css'

export default function Sidebar({ sections }) {
  const [expanded, setExpanded] = useState(true)
  const sectionIds = sections.map((s) => s.id)
  const activeId = useScrollSpy(sectionIds)

  const handleClick = (id) => {
    const el = document.getElementById(id)
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' })
      history.replaceState(null, '', `#${id}`)
    }
  }

  return (
    <nav className="sidebar">
      <div className="sidebar-title">
        <span>Table of Contents</span>
        <button
          className="sidebar-toggle"
          onClick={() => setExpanded((v) => !v)}
          aria-label={expanded ? 'Collapse table of contents' : 'Expand table of contents'}
        >
          <i className={`fa-solid fa-chevron-${expanded ? 'down' : 'right'}`} />
        </button>
      </div>
      {expanded && <ul className="sidebar-list">
        {sections.map((s) => (
          <li key={s.id}>
            <button
              className={`sidebar-link ${s.level === 2 ? 'sidebar-sub' : ''} ${
                activeId === s.id ? 'active' : ''
              }`}
              onClick={() => handleClick(s.id)}
            >
              {s.title}
            </button>
          </li>
        ))}
      </ul>}
    </nav>
  )
}
