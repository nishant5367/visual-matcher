// src/App.jsx
import React, { useState, useEffect } from "react";
import axios from "axios";

export default function App() {
  const [file, setFile] = useState(null);
  const [urlInput, setUrlInput] = useState("");
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [catalogSample, setCatalogSample] = useState([]);

  const backend = import.meta.env.VITE_API || "http://127.0.0.1:8000";

  useEffect(() => {
    // fetch a few catalog items for quick display / debugging
    axios.get(`${backend}/products`)
      .then(res => {
        if (res.data && res.data.products) {
          setCatalogSample(res.data.products.slice(0, 6));
        }
      })
      .catch(() => {});
  }, []);

  const handleFile = (e) => {
    const f = e.target.files?.[0] ?? null;
    setFile(f);
    setPreview(f ? URL.createObjectURL(f) : null);
    setResults([]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setResults([]);
    try {
      const form = new FormData();
      if (file) form.append("file", file);
      else form.append("image_url", urlInput);
      form.append("top_k", 8);
      const res = await axios.post(`${backend}/match`, form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResults(res.data.results || []);
    } catch (err) {
      // friendly error
      const msg = err?.response?.data?.error || err.message || "Request failed";
      alert("Error: " + msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ padding: 20, maxWidth: 1000, margin: "0 auto", fontFamily: "Inter, Arial" }}>
      <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 18 }}>
        <h1 style={{ margin: 0 }}>Visual Product Matcher</h1>
        <div style={{ fontSize: 13, color: "#666" }}>Demo • Local</div>
      </header>

      <form onSubmit={handleSubmit} style={{ marginBottom: 18 }}>
        <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
          <input type="file" accept="image/*" onChange={handleFile} />
          <div style={{ flex: 1 }}>
            <input
              placeholder="Or paste image URL"
              value={urlInput}
              onChange={(e) => setUrlInput(e.target.value)}
              style={{ width: "100%", padding: 8, borderRadius: 6, border: "1px solid #ddd" }}
            />
          </div>
          <button type="submit" disabled={loading} style={{ padding: "8px 14px" }}>
            {loading ? "Searching..." : "Find Similar"}
          </button>
        </div>
      </form>

      {preview && (
        <section style={{ marginBottom: 18 }}>
          <strong>Uploaded preview</strong>
          <div style={{ marginTop: 8 }}>
            <img src={preview} alt="preview" style={{ maxWidth: 260, borderRadius: 8, objectFit: "cover" }} />
          </div>
        </section>
      )}

      <section style={{ marginBottom: 22 }}>
        <strong>Results</strong>
        <div style={{ marginTop: 10, display: "flex", gap: 12, flexWrap: "wrap" }}>
          {results.length === 0 && !loading && <div style={{ color: "#666" }}>No results yet — try upload or paste URL.</div>}
          {results.map((r) => (
            <div key={r.id + "-" + r.score} style={{ width: 180, border: "1px solid #eee", borderRadius: 8, padding: 8 }}>
              <img src={r.image} alt={r.name} style={{ width: "100%", height: 120, objectFit: "cover", borderRadius: 6 }} />
              <div style={{ marginTop: 8 }}>
                <div style={{ fontWeight: 600 }}>{r.name}</div>
                <div style={{ fontSize: 12, color: "#666" }}>{r.category}</div>
                <div style={{ fontSize: 12, marginTop: 6 }}>Score: {r.score?.toFixed(3)}</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section style={{ marginTop: 20 }}>
        <strong>Sample catalog (for dev)</strong>
        <div style={{ display: "flex", gap: 10, marginTop: 8, flexWrap: "wrap" }}>
          {catalogSample.map(p => (
            <div key={p.id} style={{ width: 140, border: "1px dashed #eee", padding: 6, borderRadius: 6 }}>
              <img src={p.image} alt={p.name} style={{ width: "100%", height: 88, objectFit: "cover", borderRadius: 6 }} />
              <div style={{ fontSize: 12, marginTop: 6 }}>{p.name}</div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
