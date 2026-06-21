import { useState } from "react";
import ScoreCard from "./ScoreCard";

export default function ResultPage({ data }) {
  const [show, setShow] = useState(false);
  const a = data.analytics;
  const delta = (a.improved.score - a.original.score).toFixed(3);

  return (
    <div className="page result-page">
      <h2>Improved video</h2>
      <video className="main-video" src={data.result.video} controls playsInline />

      {!show ? (
        <button className="btn-primary" onClick={() => setShow(true)}>
          Show brain analytics →
        </button>
      ) : (
        <>
          <div className="ba2">
            <div className="ba2-col">
              <div className="ba2-label">Original</div>
              <video src={a.original.brainVideo} autoPlay loop muted playsInline />
              <img src={a.original.retentionChart} alt="retention (original)" />
              <ScoreCard value={a.original.score} label="retention" />
            </div>
            <div className="ba2-arrow">▸</div>
            <div className="ba2-col">
              <div className="ba2-label">Improved</div>
              <video src={a.improved.brainVideo} autoPlay loop muted playsInline />
              <img src={a.improved.retentionChart} alt="retention (improved)" />
              <ScoreCard value={a.improved.score} label="retention" delta={"+" + delta} />
            </div>
          </div>
          <div className="delta-badge">
            retention +{delta} ({a.original.score} → {a.improved.score})
          </div>
        </>
      )}
    </div>
  );
}
