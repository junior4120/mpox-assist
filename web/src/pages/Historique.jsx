import { useEffect, useState } from "react";
import { getResults } from "../api/client";

const classColors = {
  Mpox: "bg-red-50 text-red-800",
  Varicelle: "bg-amber-50 text-amber-800",
  "Peau saine": "bg-green-50 text-green-800",
  "Autres affections cutanées": "bg-gray-100 text-gray-700",
  Herpès: "bg-amber-50 text-amber-800",
};

export default function Historique() {
  const [results, setResults] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    getResults()
      .then(setResults)
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="bg-red-50 text-red-800 p-4 rounded-lg text-sm">
        Impossible de contacter l'API : {error}
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-semibold text-gray-800">Historique des analyses</h2>

      <div className="bg-white rounded-xl border border-gray-200 overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr className="text-left text-gray-500">
              <th className="px-4 py-3 font-medium">Date</th>
              <th className="px-4 py-3 font-medium">Référence</th>
              <th className="px-4 py-3 font-medium">Prédiction</th>
              <th className="px-4 py-3 font-medium">Confiance</th>
              <th className="px-4 py-3 font-medium">Modèle</th>
              <th className="px-4 py-3 font-medium">Synchronisé</th>
            </tr>
          </thead>
          <tbody>
            {results.map((r) => (
              <tr key={r.id} className="border-t border-gray-100">
                <td className="px-4 py-3 text-gray-500">
                  {new Date(r.created_at).toLocaleString("fr-FR")}
                </td>
                <td className="px-4 py-3 text-gray-700">{r.patient_reference}</td>
                <td className="px-4 py-3">
                  <span
                    className={`px-2 py-0.5 rounded text-xs font-semibold ${
                      classColors[r.predicted_class] || "bg-gray-100 text-gray-700"
                    }`}
                  >
                    {r.predicted_class}
                  </span>
                </td>
                <td className="px-4 py-3 text-gray-500">{(r.confidence * 100).toFixed(0)}%</td>
                <td className="px-4 py-3 text-gray-500">
                  {r.model_name} {r.model_version}
                </td>
                <td className="px-4 py-3">
                  {r.synced ? (
                    <span className="text-green-700">✓</span>
                  ) : (
                    <span className="text-gray-400">—</span>
                  )}
                </td>
              </tr>
            ))}
            {results.length === 0 && (
              <tr>
                <td colSpan={6} className="px-4 py-6 text-center text-gray-400">
                  Aucune analyse enregistrée.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}