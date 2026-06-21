import { useState, useCallback, useEffect } from "react";
import { PAGES } from "../data/pages";

export function usePages() {
  const [i, setI] = useState(0);

  const next = useCallback(() => setI((x) => Math.min(x + 1, PAGES.length - 1)), []);
  const prev = useCallback(() => setI((x) => Math.max(x - 1, 0)), []);
  const goTo = useCallback((n) => setI(Math.max(0, Math.min(n, PAGES.length - 1))), []);

  // arrow-key nav — but never hijack typing in the speech/context box
  useEffect(() => {
    const onKey = (e) => {
      const tag = (e.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea" || e.target.isContentEditable) return;
      if (e.code === "ArrowRight") {
        e.preventDefault();
        next();
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        prev();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [next, prev]);

  return { page: PAGES[i], index: i, total: PAGES.length, next, prev, goTo };
}
