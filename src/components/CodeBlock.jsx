import { useMemo, useState } from 'react'
import './CodeBlock.css'

const EXT_TO_LANG = {
  py: 'python',
  js: 'javascript',
  jsx: 'javascript',
  ts: 'typescript',
  tsx: 'typescript',
  sh: 'bash',
  bash: 'bash',
  json: 'json',
  yaml: 'yaml',
  yml: 'yaml',
  css: 'css',
  html: 'html',
  sql: 'sql',
  cpp: 'cpp',
  c: 'c',
  java: 'java',
  rs: 'rust',
  go: 'go',
  rb: 'ruby',
  md: 'markdown',
}

/**
 * Split highlighted HTML by newlines while preserving span context.
 * When hljs wraps a token across lines (e.g. a multi-line string),
 * splitting by \n breaks the <span> tags. This function tracks open
 * spans and closes/reopens them at line boundaries.
 */
function splitHighlightedLines(html) {
  const rawLines = html.split('\n')
  const result = []
  let openTags = []

  for (const line of rawLines) {
    const prefix = openTags.join('')

    const tagRegex = /<\/?span[^>]*>/g
    let m
    while ((m = tagRegex.exec(line)) !== null) {
      if (m[0].startsWith('</')) {
        openTags.pop()
      } else {
        openTags.push(m[0])
      }
    }

    const suffix = '</span>'.repeat(openTags.length)
    result.push(prefix + line + suffix)
  }

  return result
}

export default function CodeBlock({ filename, language, children }) {
  const [copied, setCopied] = useState(false)

  const resolvedLang = useMemo(() => {
    if (language) return language
    if (!filename) return null
    const ext = filename.split('.').pop()?.toLowerCase()
    return ext ? (EXT_TO_LANG[ext] || ext) : null
  }, [filename, language])

  const label = filename || resolvedLang

  const lines = useMemo(() => {
    if (!window.hljs || !resolvedLang) return null
    try {
      const highlighted = window.hljs.highlight(children, { language: resolvedLang })
      return splitHighlightedLines(highlighted.value)
    } catch {
      return null
    }
  }, [children, resolvedLang])

  const handleCopy = () => {
    navigator.clipboard.writeText(children).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  // dangerouslySetInnerHTML is used with highlight.js output from
  // developer-authored code strings (not user input), so XSS risk is controlled.
  return (
    <div className="codeblock-container">
      <div className="codeblock-header">
        {label && <span className="codeblock-lang">{label}</span>}
        <button className="codeblock-copy" onClick={handleCopy} aria-label="Copy code">
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>
      <pre>
        {lines ? (
          <table className="code-table">
            <tbody>
              {lines.map((line, i) => (
                <tr key={i}>
                  <td className="line-number">{i + 1}</td>
                  <td
                    className="line-content"
                    dangerouslySetInnerHTML={{ __html: line || '\n' }}
                  />
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <code>{children}</code>
        )}
      </pre>
    </div>
  )
}
