import { useNumbering } from '../context/NumberingContext'
import './Ref.css'

export default function Ref({ label }) {
  const { getLabel } = useNumbering()
  const entry = getLabel(label)

  if (!entry) {
    return <span style={{ color: 'red', fontWeight: 'bold' }}>[??]</span>
  }

  const prefix = entry.type === 'figure' ? 'Figure' : 'Table'

  return (
    <a href={`#${label}`} className="ref-link">
      {prefix} {entry.number}
    </a>
  )
}
