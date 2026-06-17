/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0f172a",
        surface: "#1e293b",
        surface2: "#334155",
        border: "#475569",
        muted: "#94a3b8",
        elite: "#22c55e",
        mid: "#f59e0b",
        low: "#ef4444",
        playerA: "#38bdf8",
        playerB: "#fb923c",
        pitch: "#166534",
        pitchLine: "#dcfce7"
      }
    }
  },
  plugins: []
};
