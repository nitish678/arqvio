import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./theme/tokens.css";
import "./theme/ws.css";
import "./theme/deck.css";

const rootEl = document.getElementById("root");
if (!rootEl) {
  throw new Error("ARQVIO: #root element not found in index.html");
}

ReactDOM.createRoot(rootEl).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
