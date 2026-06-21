import { useEffect, useState } from "react";

// Big engagement number with a quick count-up animation.
export default function ScoreCard({ value, label, delta }) {
  const [n, setN] = useState(0);

  useEffect(() => {
    let raf;
    let start;
    const dur = 900;
    const step = (t) => {
      if (start === undefined) start = t;
      const p = Math.min((t - start) / dur, 1);
      setN(value * p);
      if (p < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [value]);

  return (
    <div className="scorecard">
      <div className="sc-value">
        {n >= 0 ? "+" : ""}
        {n.toFixed(3)}
      </div>
      <div className="sc-label">{label}</div>
      {delta && <div className="sc-delta">▲ {delta}</div>}
    </div>
  );
}
