import './Header.css'

export default function Header({ title, authors, affiliations, links }) {
  return (
    <header className="paper-header">
      <h1 className="paper-title" style={{ whiteSpace: 'pre-line' }}>{title}</h1>

      <div className="author-list">
        {authors.map((author, i) => (
          <span key={i} className="author">
            {author.url ? (
              <a href={author.url} target="_blank" rel="noopener noreferrer">
                {author.name}
              </a>
            ) : (
              author.name
            )}
            {author.affiliations && (
              <sup>{author.affiliations.join(',')}</sup>
            )}
            {author.equal && <span className="equal-contrib"><sup>=</sup></span>}
            {i < authors.length - 1 && ', '}
          </span>
        ))}
      </div>

      {authors.some((a) => a.equal) && (
        <p className="equal-note"><sup>=</sup> Equal contribution</p>
      )}

      <div className="affiliation-list">
        {affiliations.map((aff, i) => (
          <span key={i} className="affiliation">
            <sup>{i + 1}</sup> {aff}
            {i < affiliations.length - 1 && ' \u00A0 '}
          </span>
        ))}
      </div>

      {links && links.length > 0 && (
        <div className="paper-links">
          {links.map((link, i) => (
            <a
              key={i}
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              className="paper-link"
            >
              {link.icon && <i className={link.icon}></i>}
              <span>{link.label}</span>
            </a>
          ))}
        </div>
      )}
    </header>
  )
}
