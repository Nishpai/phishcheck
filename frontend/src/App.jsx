import { useState } from "react";

const API_URL = "http://127.0.0.1:8741";

// The first three come from the dataset. The milk one is made up.
const SAMPLES = [
  {
    title: "Prize text",
    text: "You have WON a guaranteed £1000 cash or a £2000 prize. To claim yr prize call our customer service representative on 08714712394 between 10am-7pm",
  },
  {
    title: "Free tickets",
    text: "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
  },
  { title: "Call later", text: "Sorry, I'll call later" },
  { title: "Milk run", text: "Can you pick up milk on the way home? I'll be there by 7." },
];

export default function App() {
  const [text, setText] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function checkMessage(event) {
    event.preventDefault();
    setResult(null);
    setError("");
    setLoading(true);
    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      const body = await response.json();
      if (response.ok) {
        setResult(body);
      } else {
        setError(body.detail);
      }
    } catch {
      setError(`Can't reach the API at ${API_URL}. Is uvicorn running?`);
    }
    setLoading(false);
  }

  return (
    <main>
      <h1>PhishCheck</h1>
      <p>Paste an SMS and it gets labeled phish or legit.</p>
      <p className="note">Trained on old SMS spam, not modern email phishing.</p>

      <div className="samples">
        {SAMPLES.map((sample) => (
          <button key={sample.title} type="button" onClick={() => setText(sample.text)}>
            {sample.title}
          </button>
        ))}
      </div>

      <form onSubmit={checkMessage}>
        <textarea
          rows={6}
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Paste an SMS here"
        />
        <button type="submit" disabled={loading}>
          {loading ? "Checking..." : "Check message"}
        </button>
      </form>

      {error && <p className="error">{error}</p>}

      {result && (
        <div className={`result ${result.label}`}>
          <h2>{result.label === "phish" ? "Phish" : "Legit"}</h2>
          <p>{(result.phish_probability * 100).toFixed(1)}% chance of phish</p>
          <p>Words that pushed toward phish:</p>
          <ul className="words">
            {result.top_words.map((item) => (
              <li key={item.word}>
                {item.word} +{item.contribution.toFixed(3)}
              </li>
            ))}
          </ul>
        </div>
      )}
    </main>
  );
}
