/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        lab: {
          950: "#111827",
          900: "#1a2332",
          850: "#1e293b",
          800: "#243044",
          750: "#2a3548",
          700: "#334155",
          accent: "#34d399",
          glow: "#22d3ee",
        },
        ink: {
          DEFAULT: "#F1F5F9",
          muted: "#94A3B8",
          faint: "#64748B",
          line: "#334155",
        },
        status: {
          ok: "#34d399",
          "ok-bg": "#064E3B33",
          "ok-border": "#34d39955",
          warn: "#F87171",
          "warn-bg": "#7F1D1D33",
          "warn-border": "#F8717155",
        },
      },
      fontFamily: {
        sans: ["Plus Jakarta Sans", "system-ui", "sans-serif"],
        serif: ["Source Serif 4", "Georgia", "serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      boxShadow: {
        soft: "0 1px 3px rgba(28, 25, 23, 0.06), 0 4px 16px rgba(28, 25, 23, 0.04)",
        card: "0 1px 2px rgba(28, 25, 23, 0.05), inset 0 1px 0 rgba(255,255,255,0.8)",
      },
      animation: {
        float: "float 8s ease-in-out infinite",
        "float-delayed": "float 8s ease-in-out 2s infinite",
        scan: "scan 3s linear infinite",
        shimmer: "shimmer 2.5s linear infinite",
        "cell-pulse": "cellPulse 3s ease-in-out infinite",
        "spin-slow": "spin 12s linear infinite",
        "scan-vertical": "scanVertical 5s linear infinite",
        marquee: "marquee 40s linear infinite",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0) translateX(0)" },
          "50%": { transform: "translateY(-12px) translateX(6px)" },
        },
        scan: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100vh)" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        cellPulse: {
          "0%, 100%": { opacity: "0.5", transform: "scale(1)" },
          "50%": { opacity: "0.85", transform: "scale(1.05)" },
        },
        scanVertical: {
          "0%": { top: "-2%" },
          "100%": { top: "102%" },
        },
        marquee: {
          "0%": { transform: "translateX(0)" },
          "100%": { transform: "translateX(-50%)" },
        },
      },
      backgroundImage: {
        "grid-lab":
          "linear-gradient(rgba(34,211,238,0.04) 1px, transparent 1px), linear-gradient(90deg, rgba(34,211,238,0.04) 1px, transparent 1px)",
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
      },
      backgroundSize: {
        grid: "48px 48px",
      },
    },
  },
  plugins: [],
};
