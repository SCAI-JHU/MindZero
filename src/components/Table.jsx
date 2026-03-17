import { useId, useEffect } from 'react'
import { useNumbering } from '../context/NumberingContext'
import './Table.css'

export default function Table({ caption, headers, rows, label }) {
  const { registerTable, registerLabel } = useNumbering()
  const autoId = useId()
  const key = label || autoId
  const num = registerTable(key)

  useEffect(() => {
    if (label) registerLabel(label, 'table', num)
  }, [label, num, registerLabel])

  return (
    <div className="table-container" id={label || undefined}>
      <div className="table-caption">
        {label && (
          <a href={`#${label}`} className="heading-anchor" aria-label="Link to table">#&nbsp;</a>
        )}
        <strong>Table {num}.</strong> {caption}
      </div>
      <table className="three-line-table">
        <thead>
          <tr>
            {headers.map((h, i) => (
              <th key={i}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => (
                <td key={j}>{cell}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
