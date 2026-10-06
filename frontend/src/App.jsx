import { useState } from "react";

// FastAPI (api.py) listens here. This page is served from port 8742.
const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8741";

// Short texts a demo can drop into the box. Spam from the UCI file is "phish".
const SAMPLES = [
  {
    title: "Prize text",
    text: "You have WON a guaranteed £1000 cash or a £2000 prize. To claim yr prize call our customer service representative on 08714712394 between 10am-7pm",
  },
  {
    title: "Free tickets",
    text: "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
  },
  {
    title: "Call later",
    text: "Sorry, I'll call later",
  },
  {
    title: "Milk run",
    text: "Can you pick up milk on the way home? I'll be there by 7.",
  },
];

function readDetail(body) {
  if (!body) return "";
  if (typeof body.detail === "string") return body.detail;
  if (Array.isArray(body.detail) && body.detail[0]?.msg) return body.detail[0].msg;
  return "";
}

function Result({ result }) {
  const phish = result.label === "phish";
  const title = phish ? "Phish" : result.label === "legit" ? "Legit" : result.label;
  const words = result.top_words ?? [];
  const percent = Math.max(0, Math.min(100, result.phish_probability * 100));
  const percentLabel = `${percent.toFixed(1)}%`;

  return (
    <div className={phish ? "verdict phish" : "verdict legit"}>
      <div className="verdict-top">
        <p className="verdict-label">{title}</p>
        <p className="score">
          <span className="score-value">{percentLabel}</span>
          <span className="score-caption">chance of phish</span>
        </p>
      </div>
      <div
        className="meter"
        role="meter"
        aria-label="Chance of phish"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={Number(percent.toFixed(1))}
      >
        <span style={{ width: `${percent}%` }} />
      </div>
      <h2>Words that pushed toward phish</h2>
      {words.length === 0 ? (
        <p className="hint">No word in this message had a positive push toward phish.</p>
      ) : (
        <ul className="words">
          {words.map((item) => (
            <li key={item.word}>
              <span className="word">{item.word}</span>
              <span className="weight">+{Number(item.contribution).toFixed(3)}</span>
            </li>
          ))}
        </ul>
      )}
      <p className="formula">
        Contribution = tf-idf in this message × the word's phish coefficient. The
        percent uses every word, not only this list.
      </p>
    </div>
  );
}

export default function App() {
  const [draft, setDraft] = useState("");
  const [status, setStatus] = useState("idle");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  async function onSubmit(event) {
    event.preventDefault();
    const message = draft.trim();
    if (!message) {
      setStatus("error");
      setError("Paste an SMS or short email first.");
      setResult(null);
      return;
    }
    if (status === "loading") return;

    setStatus("loading");
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: message }),
      });
      const body = await response.json().catch(() => null);
      if (!response.ok) {
        throw new Error(readDetail(body) || `The API returned ${response.status}.`);
      }
      setResult(body);
      setStatus("done");
    } catch (err) {
      setResult(null);
      setStatus("error");
      // fetch throws TypeError when the API process is not running.
      if (err instanceof TypeError) {
        setError(
          `Can't reach the API at ${API_URL}. From the repo root run: uvicorn api:app --host 127.0.0.1 --port 8741`,
        );
      } else {
        setError(err.message);
      }
    }
  }

  function fillSample(sample) {
    setDraft(sample);
    setResult(null);
    setError("");
    setStatus("idle");
  }

  return (
    <main className="page">
      <header>
        <p className="eyebrow">SMS check</p>
        <h1>PhishCheck</h1>
        <p className="lede">
          Paste an SMS. Spam is labeled <strong>phish</strong> and an ordinary
          message is labeled <strong>legit</strong>. This is old SMS spam, not a
          check for modern email phishing.
        </p>
      </header>

      <form className="checker" onSubmit={onSubmit}>
        <p className="samples-label">Try a sample</p>
        <div className="sample-row">
          {SAMPLES.map((sample) => (
            <button
              key={sample.title}
              type="button"
              className={draft === sample.text ? "sample active" : "sample"}
              aria-pressed={draft === sample.text}
              onClick={() => fillSample(sample.text)}
            >
              {sample.title}
            </button>
          ))}
        </div>

        <label className="field-label" htmlFor="message">
          Message
        </label>
        <textarea
          id="message"
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          rows={6}
          placeholder="Paste an SMS here"
        />
        <button className="check" type="submit" disabled={status === "loading"}>
          {status === "loading" ? "Checking…" : "Check message"}
        </button>
      </form>

      <section className="result" aria-live="polite" aria-busy={status === "loading"}>
        {status === "idle" && (
          <p className="hint">The label, chance of phish, and top words will show up here.</p>
        )}
        {status === "loading" && <p className="loading">Checking the message…</p>}
        {status === "error" && <p className="error">{error}</p>}
        {status === "done" && result && <Result result={result} />}
      </section>

      <p className="footnote">
        Trained on the UCI SMS Spam Collection. These are old phone texts, not
        modern email phishing.
      </p>
    </main>
  );
}
