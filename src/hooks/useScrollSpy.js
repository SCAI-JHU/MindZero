import { useState, useEffect } from 'react'

export function useScrollSpy(sectionIds) {
  const [activeId, setActiveId] = useState('')

  useEffect(() => {
    const elements = sectionIds
      .map((id) => document.getElementById(id))
      .filter(Boolean)

    if (elements.length === 0) return

    const READING_LINE = 100
    let ticking = false

    const updateActive = () => {
      let active = null
      for (const id of sectionIds) {
        const el = document.getElementById(id)
        if (el && el.getBoundingClientRect().top <= READING_LINE) {
          active = id
        }
      }
      if (active) setActiveId(active)
      ticking = false
    }

    const onScroll = () => {
      if (!ticking) {
        requestAnimationFrame(updateActive)
        ticking = true
      }
    }

    window.addEventListener('scroll', onScroll, { passive: true })
    updateActive()
    return () => window.removeEventListener('scroll', onScroll)
  }, [sectionIds])

  return activeId
}
