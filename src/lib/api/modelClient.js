const baseUrl = process.env.MODEL_API_BASE_URL || "http://localhost:8000";

export async function modelRequest(path, options = {}) {
  const response = await fetch(`${baseUrl}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    cache: "no-store"
  });

  if (!response.ok) {
    throw new Error(`Model service returned ${response.status}`);
  }

  return response.json();
}

export function isDevelopment() {
  return process.env.NODE_ENV !== "production";
}

export function modelError(message = "Model service is unavailable") {
  return { error: message };
}
