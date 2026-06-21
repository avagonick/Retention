import { useState } from "react";
import SpeechInput from "./SpeechInput";

export default function UploadPage({ data, onScore }) {
  const [context, setContext] = useState("");

  return (
    <div className="page upload-page">
      <h2>Original video</h2>
      <video className="main-video" src={data.original.video} controls playsInline />
      <SpeechInput
        value={context}
        onChange={setContext}
        placeholder={data.contextPlaceholder}
      />
      <button className="btn-primary" onClick={onScore}>
        Score with TRIBE v2 →
      </button>
    </div>
  );
}
