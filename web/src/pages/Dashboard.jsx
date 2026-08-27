import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { getStatistics, getResults } from "../api/client";

const classColors = {
  Mpox: "bg-red-50 text-red-800",
  Varicelle: "bg-amber-50 text-amber-800",
  "Peau saine": "bg-green-50 text-green-800",
  "Autres affections cutanées": "bg-gray-100 text-gray-700",
  Herpès: "bg-amber-50 text-amber-800",
};

export default function Dashboard() {
  const [stats, setStats] = useState(null);
  const [recent, setRecent] = useState([]);
  const [error, setError] = useState(null);

  useEffect(() => {
    Promise.all([getStatistics(), getResults()])
      .then(([statsData, resultsData]) => {
        setStats(statsData);
        setRecent(resultsData.slice(0, 6));
      })
      .catch((err) => setError(err.message));
  }, []);

  if (error) {
    return (
      <div className="bg-red-50 text-red-800 p-4 rounded-lg text-sm">
        Impossible de contacter l'API : {error}. Vérifie que le backend tourne sur le port 8000.
      </div>
    );
  }

  if (!stats) {
    return <p className="text-gray-500 text-sm">Chargement du tableau de bord…</p>;
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-800">Dashboard de supervision</h2>
        {stats.is_demo_data && (
          <span className="bg-amber-50 text-amber-800 text-xs font-semibold px-3 py-1 rounded-md">
            Données de démonstration
          </span>
        )}
        <Link
          to="/analyse"
          className="bg-blue-800 text-white text-sm px-4 py-2 rounded-lg hover:bg-blue-900"
        >
          + Nouvelle analyse
        </Link>
      </div>

      <div className="grid grid-cols-4 gap-4">
        <StatCard label="Total analyses" value={stats.total_analyses} />
        <StatCard label="Cas suspects" value={stats.suspect_cases} />
        <StatCard label="Modèle" value={stats.model_name} small />
        <StatCard
          label="Temps moyen d'inférence"
          value={
            stats.avg_inference_time_ms != null
              ? `${stats.avg_inference_time_ms.toFixed(1)} ms`
              : "—"
          }
        />
      </div>

      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="text-sm font-semibold text-gray-600 mb-3">Répartition des classes</h3>
          <div className="space-y-2">
            {Object.entries(stats.class_distribution).map(([cls, count]) => (
              <div key={cls} className="flex items-center gap-3 text-sm">
                <span className="w-40 text-gray-600">{cls}</span>
                <div className="flex-1 bg-gray-100 rounded h-2">
                  <div
                    className="bg-blue-700 h-2 rounded"
                    style={{
                      width: stats.total_analyses
                        ? `${(count / stats.total_analyses) * 100}%`
                        : "0%",
                    }}
                  />
                </div>
                <span className="text-gray-500 w-8 text-right">{count}</span>
              </div>
            ))}
          </div>
        </div>

        <div className="bg-white rounded-xl border border-gray-200 p-5">
          <h3 className="text-sm font-semibold text-gray-600 mb-3">Historique récent</h3>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 border-b border-gray-100">
                <th className="pb-2 font-medium">Référence</th>
                <th className="pb-2 font-medium">Prédiction</th>
                <th className="pb-2 font-medium">Confiance</th>
              </tr>
            </thead>
            <tbody>
              {recent.map((r) => (
                <tr key={r.id} className="border-b border-gray-50">
                  <td className="py-2 text-gray-700">{r.patient_reference}</td>
                  <td className="py-2">
                    <span
                      className={`px-2 py-0.5 rounded text-xs font-semibold ${
                        classColors[r.predicted_class] || "bg-gray-100 text-gray-700"
                      }`}
                    >
                      {r.predicted_class}
                    </span>
                  </td>
                  <td className="py-2 text-gray-500">
                    {(r.confidence * 100).toFixed(0)}%
                  </td>
                </tr>
              ))}
              {recent.length === 0 && (
                <tr>
                  <td colSpan={3} className="py-4 text-center text-gray-400">
                    Aucune analyse pour le moment.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, small }) {
  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <p className="text-xs text-gray-500 mb-1">{label}</p>
      <p className={small ? "text-base font-semibold text-gray-800" : "text-2xl font-bold text-gray-800"}>
        {value}
      </p>
    </div>
  );
}