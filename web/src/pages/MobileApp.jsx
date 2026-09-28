import { useState } from "react";
import { predict, UPLOADS_BASE } from "../api/client";

const classColors = {
  Mpox: "bg-blue-700",
  Varicelle: "bg-amber-500",
  "Peau saine": "bg-green-600",
  "Autres affections cutanées": "bg-gray-400",
};

export default function MobileApp() {
  const [screen, setScreen] = useState("home"); // home | capture | result
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  function handleFileChange(e) {
    const f = e.target.files[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }

  async function handleAnalyze() {
    if (!file) return;
    setLoading(true);
    setError(null);
    try {
      const data = await predict(file);
      setResult(data);
      setScreen("result");
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  function resetToHome() {
    setScreen("home");
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  }

  return (
    <div className="min-h-screen bg-gray-100 flex justify-center">
      <div className="w-full max-w-sm bg-white min-h-screen flex flex-col">
        {/* Barre du haut, style app mobile */}
        <div className="px-5 pt-6 pb-4 border-b border-gray-100">
          <h1 className="text-lg font-bold text-blue-900">MPOX-Assist</h1>
          <p className="text-xs text-gray-500">Outil intelligent d'aide au triage</p>
        </div>

        <div className="flex-1 px-5 py-5">
          {screen === "home" && <HomeScreen onNew={() => setScreen("capture")} />}
          {screen === "capture" && (
            <CaptureScreen
              preview={preview}
              onFileChange={handleFileChange}
              onAnalyze={handleAnalyze}
              onBack={resetToHome}
              loading={loading}
              error={error}
              hasFile={!!file}
            />
          )}
          {screen === "result" && result && (
            <ResultScreen result={result} onNewAnalysis={resetToHome} />
          )}
        </div>
      </div>
    </div>
  );
}

function HomeScreen({ onNew }) {
  return (
    <div className="flex flex-col gap-3">
      <div className="bg-green-50 text-green-800 text-xs font-medium px-3 py-2 rounded-lg text-center">
        ● Mode hors ligne disponible
      </div>
      <button
        onClick={onNew}
        className="bg-blue-800 text-white py-3 rounded-xl font-medium"
      >
        Nouvelle analyse
      </button>
      <button className="border border-gray-300 text-gray-700 py-3 rounded-xl font-medium">
        Historique
      </button>
      <button className="border border-gray-300 text-gray-700 py-3 rounded-xl font-medium">
        Synchroniser
      </button>
      <p className="text-xs text-gray-400 text-center mt-2">
        Prototype expérimental d'aide au triage
      </p>
    </div>
  );
}

function CaptureScreen({ preview, onFileChange, onAnalyze, onBack, loading, error, hasFile }) {
  return (
    <div className="flex flex-col gap-3">
      <button onClick={onBack} className="text-sm text-gray-500 self-start mb-1">
        ← Retour
      </button>
      <p className="text-sm font-semibold text-gray-700">Capture</p>

      {preview ? (
        <img src={preview} alt="Aperçu" className="w-full h-48 object-cover rounded-xl border border-gray-200" />
      ) : (
        <div className="w-full h-48 bg-gray-100 rounded-xl flex items-center justify-center text-gray-400 text-sm">
          Image de la lésion
        </div>
      )}

      <label className="bg-blue-800 text-white py-3 rounded-xl font-medium text-center cursor-pointer">
        📷 Prendre une photo / Importer
        <input
          type="file"
          accept="image/jpeg,image/png"
          capture="environment"
          onChange={onFileChange}
          className="hidden"
        />
      </label>

      <p className="text-xs text-gray-500">
        Assurez-vous que la lésion est visible, nette et correctement éclairée.
      </p>

      {error && <p className="text-red-700 text-xs bg-red-50 p-2 rounded-lg">{error}</p>}

      <button
        onClick={onAnalyze}
        disabled={!hasFile || loading}
        className="bg-green-700 text-white py-3 rounded-xl font-medium disabled:bg-gray-300 mt-2"
      >
        {loading ? "Analyse en cours…" : "Analyser"}
      </button>
    </div>
  );
}

function ResultScreen({ result, onNewAnalysis }) {
  return (
    <div className="flex flex-col gap-3">
      <span className="bg-amber-50 text-amber-800 text-xs font-semibold px-3 py-1 rounded-md self-start">
        {result.ai_mode === "demo" ? "MODE DÉMONSTRATION" : "MODÈLE RÉEL"}
      </span>

      <p className="text-sm text-gray-500">Classe estimée</p>
      <p className="text-xl font-bold text-gray-800 -mt-2">{result.predicted_class}</p>

      <div className="space-y-1.5">
        {Object.entries(result.probabilities)
          .sort((a, b) => b[1] - a[1])
          .map(([cls, prob]) => (
            <div key={cls} className="text-xs">
              <div className="flex justify-between mb-0.5">
                <span className="text-gray-600">{cls}</span>
                <span className="text-gray-500">{(prob * 100).toFixed(0)}%</span>
              </div>
              <div className="bg-gray-100 rounded h-1.5">
                <div
                  className={`h-1.5 rounded ${classColors[cls] || "bg-gray-400"}`}
                  style={{ width: `${prob * 100}%` }}
                />
              </div>
            </div>
          ))}
      </div>

      <div className="grid grid-cols-3 gap-1.5 mt-1">
        <img src={`${UPLOADS_BASE}${result.original_image_url}`} className="rounded-lg h-16 w-full object-cover" alt="Originale" />
        <img src={`${UPLOADS_BASE}${result.heatmap_url}`} className="rounded-lg h-16 w-full object-cover" alt="Heatmap" />
        <img src={`${UPLOADS_BASE}${result.overlay_url}`} className="rounded-lg h-16 w-full object-cover" alt="Superposition" />
      </div>

      <div className="bg-blue-50 text-blue-900 text-xs p-3 rounded-lg leading-relaxed mt-1">
        {result.triage_message}
      </div>

      <button
        onClick={onNewAnalysis}
        className="border border-gray-300 text-gray-700 py-3 rounded-xl font-medium mt-2"
      >
        Nouvelle analyse
      </button>
    </div>
  );
}