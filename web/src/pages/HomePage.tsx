import { Link } from 'react-router-dom'

const steps = ['Raw input', 'Representation', 'Preprocessing', 'Saved model', 'Prediction']

export function HomePage() {
  return (
    <div className="page home-page">
      <section className="hero">
        <div>
          <span className="eyebrow">Assignment 02 · Intelligent System Development</span>
          <h1>Three models.<br /><em>One deployable workflow.</em></h1>
          <p>Follow raw data through numerical representation, fitted models, evaluation and local web/mobile inference.</p>
          <div className="hero-actions">
            <Link className="primary-button" to="/diabetes">Try a prediction</Link>
            <Link className="text-link" to="/about">View methodology →</Link>
          </div>
        </div>
        <div className="hero-panel" aria-label="System pipeline">
          {steps.map((step, index) => (
            <div className="flow-step" key={step}>
              <span>{String(index + 1).padStart(2, '0')}</span><strong>{step}</strong>
              {index < steps.length - 1 && <i>↓</i>}
            </div>
          ))}
        </div>
      </section>

      <section className="systems-section">
        <div className="section-heading"><span>Available systems</span><h2>Choose a model to explore</h2></div>
        <div className="system-grid">
          <article className="system-card teal-card">
            <span className="system-number">01</span><span className="task-pill">Classification</span>
            <h3>Diabetes<br />Prediction</h3>
            <p>Six raw patient attributes feed a fitted Random Forest pipeline. Output is an educational class prediction, never a diagnosis.</p>
            <div className="card-links"><Link to="/diabetes">Open system →</Link><Link to="/diabetes/knowledge-graph">View graph</Link></div>
          </article>
          <article className="system-card amber-card">
            <span className="system-number">02</span><span className="task-pill">Regression</span>
            <h3>Vietnam House<br />Price Prediction</h3>
            <p>Eleven raw property attributes pass through numerical and categorical preprocessing before a price estimate in billion VND.</p>
            <div className="card-links"><Link to="/house-price">Open system →</Link></div>
          </article>
          <article className="system-card teal-card">
            <span className="system-number">03</span><span className="task-pill">Text classification</span>
            <h3>Customer<br />Preference</h3>
            <p>Review text becomes a sparse 12,000-term TF-IDF vector, joined with five behavioral features for preference prediction.</p>
            <div className="card-links"><Link to="/ecommerce">Open system →</Link></div>
          </article>
        </div>
      </section>
    </div>
  )
}
