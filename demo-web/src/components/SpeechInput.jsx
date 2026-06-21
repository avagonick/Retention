import { useRef, useState, useEffect } from "react";

// Browser Web Speech API for live dictation (zero-config, no key).
// Swap for Deepgram streaming STT in production (needs a key + proxy).
export default function SpeechInput({ value, onChange, placeholder }) {
  const [listening, setListening] = useState(false);
  const [supported, setSupported] = useState(true);
  const recRef = useRef(null);

  useEffect(() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      setSupported(false);
      return;
    }
    const rec = new SR();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = "en-US";
    rec.onresult = (e) => {
      let text = "";
      for (let i = 0; i < e.results.length; i++) text += e.results[i][0].transcript;
      onChange(text);
    };
    rec.onend = () => setListening(false);
    recRef.current = rec;
    return () => rec.abort();
  }, [onChange]);

  const toggle = () => {
    const rec = recRef.current;
    if (!rec) return;
    if (listening) {
      rec.stop();
      setListening(false);
    } else {
      try {
        rec.start();
        setListening(true);
      } catch {
        /* already started */
      }
    }
  };

  return (
    <div className="speech">
      <div className="speech-row">
        <button
          className={"mic" + (listening ? " on" : "")}
          onClick={toggle}
          disabled={!supported}
        >
          {listening ? "● recording…" : "🎤 speak"}
        </button>
        <span className="speech-label">add context for the model (optional)</span>
      </div>
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        rows={3}
      />
      {!supported && (
        <div className="warn-note">
          Speech API unavailable in this browser — type instead. (Deepgram in production.)
        </div>
      )}
    </div>
  );
}
