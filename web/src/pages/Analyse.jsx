import { useState } from "react";
import { predict, UPLOADS_BASE } from "../api/client";

const classColors = {
  Mpox: "bg-blue-700",
  Varicelle: "bg-amber-500",
  "Peau saine": "bg-green-600",
  "Autres affections cutanées": "bg-gray-400",
  Herpès: "bg-amber-500",
};

export default function Analyse() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [patientRef, setPatientRef] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  function handleFileChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setResult(null);
    setError(null);
    setPreview(URL.createObjectURL(f));
  }

  async function handleAnalyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predict(file, patientRef || undefined);
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold text-gray-800">Nouvelle analyse</h2>

      <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-4">
        <div>
          <label className="block text-sm text-gray-600 mb-1">
            Référence patient (optionnel, anonyme)
          </label>
          <input
            type="text"
            value={patientRef}
            onChange={(e) => setPatientRef(e.target.value)}
            placeholder="ex. ANON-001"
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm w-64"
          />
        </div>

        <div>
          <label className="block text-sm text-gray-600 mb-2">Image de la lésion</label>
          <input type="file" accept="image/jpeg,image/png" onChange={handleFileChange} />
        </div>

        {preview && (
          <img src={preview} alt="Aperçu" className="h-48 rounded-lg border border-gray-200" />
        )}

        <button
          onClick={handleAnalyze}
          disabled={!file || loading}
          className="bg-blue-800 text-white text-sm px-5 py-2 rounded-lg hover:bg-blue-900 disabled:bg-gray-300"
        >
          {loading ? "Analyse en cours…" : "Analyser"}
        </button>

        {error && (
          <p className="text-red-700 text-sm bg-red-50 p-3 rounded-lg">{error}</p>
        )}
      </div>

      {result && (
        <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-5">
          <div className="flex items-center justify-between">
            <span className="bg-amber-50 text-amber-800 text-xs font-semibold px-3 py-1 rounded-md">
              {result.ai_mode === "demo" ? "MODE DÉMONSTRATION" : "MODÈLE RÉEL"}
            </span>
            <span className="text-xs text-gray-400">
              Temps d'inférence : {result.inference_time_ms.toFixed(1)} ms
            </span>
          </div>

          <div>
            <p className="text-sm text-gray-500">Classe estimée</p>
            <p className="text-2xl font-bold text-gray-800">{result.predicted_class}</p>
          </div>

          <div className="space-y-1.5">
            {Object.entries(result.probabilities)
              .sort((a, b) => b[1] - a[1])
              .map(([cls, prob]) => (
                <div key={cls} className="flex items-center gap-3 text-sm">
                  <span className="w-48 text-gray-600">{cls}</span>
                  <div className="flex-1 bg-gray-100 rounded h-2">
                    <div
                      className={`h-2 rounded ${classColors[cls] || "bg-gray-400"}`}
                      style={{ width: `${prob * 100}%` }}
                    />
                  </div>
                  <span className="text-gray-500 w-10 text-right">{(prob * 100).toFixed(0)}%</span>
                </div>
              ))}
          </div>

          <div>
            <p className="text-sm font-semibold text-gray-600 mb-2">Explicabilité (Grad-CAM)</p>
            <div className="grid grid-cols-3 gap-3">
              <ImageBox label="Image originale" src={`${UPLOADS_BASE}${result.original_image_url}`} />
              <ImageBox label="Heatmap" src={`${UPLOADS_BASE}${result.heatmap_url}`} />
              <ImageBox label="Superposition" src={`${UPLOADS_BASE}${result.overlay_url}`} />
            </div>
            <p className="text-xs text-gray-400 mt-2">{result.warning}</p>
          </div>

          <div className="bg-blue-50 text-blue-900 text-sm p-4 rounded-lg leading-relaxed">
            {result.triage_message}
          </div>
        </div>
      )}
    </div>
  );
}

function ImageBox({ label, src }) {
  return (
    <div>
      <img src={src} alt={label} className="rounded-lg border border-gray-200 w-full h-32 object-cover" />
      <p className="text-xs text-gray-400 mt-1 text-center">{label}</p>
    </div>
  );
}