import { useEffect, useState } from "react";
import { getModelInfo } from "../api/client";

export default function Modele() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getModelInfo()
      .then(setInfo)
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="bg-red-50 text-red-800 p-4 rounded-lg text-sm">
        Impossible de contacter l'API : {error}
      </div>
    );
  }

  if (!info) {
    return <p className="text-gray-500 text-sm">Chargement…</p>;
  }

  const metrics = [
    { label: "Accuracy", value: info.accuracy },
    { label: "Recall", value: info.recall },
    { label: "Precision", value: info.precision },
    { label: "F1-score", value: info.f1_score },
  ];

  return (
    <div className="space-y-6 max-w-2xl">
      <h2 className="text-lg font-semibold text-gray-800">Informations du modèle</h2>

      <div className="bg-white rounded-xl border border-gray-200 p-5 space-y-4">
        <Row label="Nom" value={info.name} />
        <Row label="Version" value={info.version} />
        <Row label="Classes" value={info.classes.join(", ")} />
        <Row label="Explicabilité" value={info.explainability} />
        <Row
          label="Mode"
          value={
            <span
              className={`px-2 py-0.5 rounded text-xs font-semibold ${
                info.mode === "demo"
                  ? "bg-amber-50 text-amber-800"
                  : "bg-green-50 text-green-800"
              }`}
            >
              {info.mode === "demo" ? "Démonstration" : "Modèle réel"}
            </span>
          }
        />
      </div>

      <div className="bg-white rounded-xl border border-gray-200 p-5">
        <h3 className="text-sm font-semibold text-gray-600 mb-3">Performances mesurées</h3>
        {info.mode === "demo" ? (
          <p className="text-sm text-gray-400 italic">
            Aucune métrique de performance en mode démonstration — aucune valeur n'est inventée.
          </p>
        ) : (
          <div className="grid grid-cols-4 gap-4">
            {metrics.map((m) => (
              <div key={m.label}>
                <p className="text-xs text-gray-500">{m.label}</p>
                <p className="text-xl font-bold text-gray-800">
                  {m.value != null ? `${(m.value * 100).toFixed(1)}%` : "—"}
                </p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function Row({ label, value }) {
  return (
    <div className="flex justify-between text-sm border-b border-gray-50 pb-2 last:border-0 last:pb-0">
      <span className="text-gray-500">{label}</span>
      <span className="text-gray-800 font-medium">{value}</span>
    </div>
  );
}