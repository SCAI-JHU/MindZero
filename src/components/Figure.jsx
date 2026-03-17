import { useId, useEffect } from 'react'
import { useNumbering } from '../context/NumberingContext'
import './Figure.css'

export default function Figure({ src, alt, caption, width, full, label }) {
  const { registerFigure, registerLabel } = useNumbering()
  const autoId = useId()
  const key = label || autoId
  const num = registerFigure(key)

  useEffect(() => {
    if (label) registerLabel(label, 'figure', num)
  }, [label, num, registerLabel])

  return (
    <figure className={`figure-container ${full ? 'figure-full' : ''}`} id={label || undefined}>
      <img
        src={src}
        alt={alt || caption || `Figure ${num}`}
        style={width ? { width } : undefined}
      />
      <figcaption className="figure-caption">
        {label && (
          <a href={`#${label}`} className="heading-anchor" aria-label="Link to figure">#&nbsp;</a>
        )}
        <strong>Figure {num}.</strong> {caption}
      </figcaption>
    </figure>
  )
}
