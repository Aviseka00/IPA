import MicroscopyShowcase from "./MicroscopyShowcase";

export default function BioBackground({ variant = "default" }: { variant?: "default" | "auth" }) {
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden bg-lab-900" aria-hidden>
      <MicroscopyShowcase variant="ambient" />
      <div className="absolute inset-0 bg-grid-lab bg-grid opacity-50" />
      <div
        className={`absolute inset-0 bg-gradient-radial via-transparent to-transparent ${
          variant === "auth" ? "from-emerald-500/5 from-30%" : "from-cyan-500/4 from-20%"
        }`}
        style={{ backgroundSize: "100% 100%" }}
      />
    </div>
  );
}
