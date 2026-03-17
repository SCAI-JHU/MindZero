import { createContext, useContext, useRef, useState, useEffect, useCallback } from 'react'

const TocContext = createContext(null)

export function TocProvider({ children }) {
  const sectionsRef = useRef(new Map())
  const [sections, setSections] = useState([])
  const pendingFlushRef = useRef(null)

  const scheduleFlush = useCallback(() => {
    if (pendingFlushRef.current) return
    pendingFlushRef.current = requestAnimationFrame(() => {
      pendingFlushRef.current = null
      const entries = Array.from(sectionsRef.current.values())
      entries.sort((a, b) => {
        const elA = document.getElementById(a.id)
        const elB = document.getElementById(b.id)
        if (!elA || !elB) return 0
        const pos = elA.compareDocumentPosition(elB)
        if (pos & Node.DOCUMENT_POSITION_FOLLOWING) return -1
        if (pos & Node.DOCUMENT_POSITION_PRECEDING) return 1
        return 0
      })
      setSections(entries)
    })
  }, [])

  const register = useCallback((id, title, level) => {
    if (!sectionsRef.current.has(id)) {
      sectionsRef.current.set(id, { id, title, level })
      scheduleFlush()
    }
    return () => {
      sectionsRef.current.delete(id)
      scheduleFlush()
    }
  }, [scheduleFlush])

  return (
    <TocContext.Provider value={{ register, sections }}>
      {children}
    </TocContext.Provider>
  )
}

export function useTocRegister(id, title, level) {
  const { register } = useContext(TocContext)
  useEffect(() => {
    return register(id, title, level)
  }, [id, title, level, register])
}

export function useTocSections() {
  return useContext(TocContext).sections
}
