import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import "./index.css";

// No StrictMode on purpose: the demo is driven by timers / typewriter effects,
// and StrictMode's double-invoke in dev would fire them twice.
ReactDOM.createRoot(document.getElementById("root")).render(<App />);
