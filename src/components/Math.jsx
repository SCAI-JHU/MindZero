import { useRef, useEffect } from 'react'

export default function Math({ display, children }) {
  const ref = useRef(null)

  useEffect(() => {
    if (!ref.current) return

    let cancelled = false

    function typeset() {
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise([ref.current]).catch(() => {})
      } else if (!cancelled) {
        setTimeout(typeset, 100)
      }
    }

    typeset()

    return () => { cancelled = true }
  }, [children])

  if (display) {
    return (
      <div ref={ref} className="math-display" style={{ margin: '1rem 0', textAlign: 'center' }}>
        {`\\[${children}\\]`}
      </div>
    )
  }

  return (
    <span ref={ref} className="math-inline">
      {`\\(${children}\\)`}
    </span>
  )
}
