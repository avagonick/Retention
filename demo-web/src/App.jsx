import { usePages } from "./hooks/usePages";
import { PAGES } from "./data/pages";
import { demoData } from "./data/demoData";
import BackgroundVideo from "./components/BackgroundVideo";
import HeroPage from "./components/HeroPage";
import UploadPage from "./components/UploadPage";
import TribeOutputPage from "./components/TribeOutputPage";
import LoopPage from "./components/LoopPage";
import ResultPage from "./components/ResultPage";

export default function App() {
  const { page, index, total, next, prev, goTo } = usePages();
  const bare = page.id === "hero";

  return (
    <>
      <BackgroundVideo />
      <div className="app">
        <header className="topbar">
          <div className="brand">
            Retention <span className="tag">demo</span>
          </div>
          <div className="dots">
            {PAGES.map((p, i) => (
              <button
                key={p.id}
                className={"dot" + (i === index ? " on" : "") + (i < index ? " done" : "")}
                onClick={() => goTo(i)}
                title={p.label}
              />
            ))}
          </div>
        </header>

        <main className={"stage" + (bare ? " bare" : "")}>
          {page.id === "hero" && <HeroPage onStart={next} />}
          {page.id === "upload" && <UploadPage data={demoData} onScore={next} />}
          {page.id === "tribe" && <TribeOutputPage data={demoData} />}
          {page.id === "loop" && <LoopPage />}
          {page.id === "result" && <ResultPage data={demoData} />}
        </main>

        <footer className="navbar">
          <button onClick={prev} disabled={index === 0}>
            ◂ Prev
          </button>
          <span className="navlabel">
            {index + 1}/{total} · {page.label}
          </span>
          <button onClick={next} disabled={index === total - 1}>
            Next ▸
          </button>
        </footer>
      </div>
    </>
  );
}
