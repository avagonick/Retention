export default function TribeOutputPage({ data }) {
  const t = data.tribe;
  return (
    <div className="page tribe-page">
      <h2>TRIBE v2 output</h2>
      <div className="array-window">
        <div className="array-head">
          preds.npy — shape ({t.shape[0]}, {t.shape[1]}) · float32
        </div>
        <div className="array-grid">
          <table>
            <thead>
              <tr>
                <th>t</th>
                {t.rows[0].map((_, c) => (
                  <th key={c}>v{c}</th>
                ))}
                <th>…</th>
              </tr>
            </thead>
            <tbody>
              {t.rows.map((row, r) => (
                <tr key={r}>
                  <td className="tcol">{r}s</td>
                  {row.map((x, c) => (
                    <td key={c} className={x >= 0 ? "pos" : "neg"}>
                      {x.toFixed(3)}
                    </td>
                  ))}
                  <td className="ell">…</td>
                </tr>
              ))}
              <tr>
                <td className="tcol">⋮</td>
                {t.rows[0].map((_, c) => (
                  <td key={c} className="ell">
                    ⋮
                  </td>
                ))}
                <td className="ell">⋱</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
