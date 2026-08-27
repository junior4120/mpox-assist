const API_BASE = "http://localhost:8000/api/v1";
const API_KEY = "poc-demo-key-change-me";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      ...(options.headers || {}),
    },
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Erreur API (${res.status}) : ${text}`);
  }
  return res.json();
}

export function getModelInfo() {
  return request("/model/info");
}

export function getStatistics() {
  return request("/statistics");
}

export function getResults() {
  return request("/results");
}

export function getResult(id) {
  return request(`/results/${id}`);
}

export async function predict(file, patientReference) {
  const formData = new FormData();
  formData.append("file", file);
  if (patientReference) {
    formData.append("patient_reference", patientReference);
  }
  const res = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: { "X-API-Key": API_KEY },
    body: formData,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`Erreur API (${res.status}) : ${text}`);
  }
  return res.json();
}

export const UPLOADS_BASE = "http://localhost:8000";