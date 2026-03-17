import './Video.css'

export default function Video({ src, type = 'youtube', caption }) {
  const isYouTube = type === 'youtube'

  return (
    <div className="video-container">
      {isYouTube ? (
        <div className="video-responsive">
          <iframe
            src={src}
            title={caption || 'Video'}
            frameBorder="0"
            allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
            allowFullScreen
          />
        </div>
      ) : (
        <video controls>
          <source src={src} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      )}
      {caption && <p className="video-caption">{caption}</p>}
    </div>
  )
}
