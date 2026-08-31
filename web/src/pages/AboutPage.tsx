export function AboutPage() {
  return (
    <div className="page about-page">
      <section className="page-heading"><span className="eyebrow">System documentation</span><h1>About the assignment</h1><p>A local, end-to-end intelligent system built around reproducible notebook experiments and trusted saved pipelines.</p></section>
      <div className="about-grid">
        <article><span>01</span><h2>Training source</h2><p>The notebooks define data quality decisions, representations, controlled experiments, final configurations and evaluation metrics.</p></article>
        <article><span>02</span><h2>Inference contract</h2><p>FastAPI loads each fitted joblib Pipeline once. Raw inputs pass directly to its learned preprocessing and estimator without refitting.</p></article>
        <article><span>03</span><h2>Knowledge layer</h2><p>Neo4j represents the Diabetes system, model, features, preprocessing steps, experiment, target and reported metrics.</p></article>
        <article><span>04</span><h2>Limitations</h2><p>All three outputs are educational demonstrations: not diagnosis, professional valuation, or independently observed customer intention.</p></article>
      </div>
      <section className="contract-strip"><strong>Representation consistency</strong><span>Training Pipeline</span><i>=</i><span>Inference Pipeline</span></section>
    </div>
  )
}
