import { useEffect, useState } from "react";
import { MICROSCOPY_IMAGES, SHOWCASE_SLIDES } from "../assets/microscopy";

type Variant = "hero" | "ambient" | "banner" | "compact";

export default function MicroscopyShowcase({ variant = "hero" }: { variant?: Variant }) {
  if (variant === "ambient") return <AmbientLayer />;
  if (variant === "banner") return <BannerStrip />;
  if (variant === "compact") return <CompactCarousel />;
  return <HeroShowcase />;
}

function HeroShowcase() {
  const [slide, setSlide] = useState(0);
  const [scanning, setScanning] = useState(true);

  useEffect(() => {
    const t = setInterval(() => setSlide((s) => (s + 1) % SHOWCASE_SLIDES.length), 6000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const t = setInterval(() => setScanning((s) => !s), 4000);
    return () => clearInterval(t);
  }, []);

  const current = SHOWCASE_SLIDES[slide];

  return (
    <div className="relative h-full w-full overflow-hidden bg-lab-850">
      <div className="absolute inset-0 bg-gradient-to-br from-slate-800/80 via-lab-850 to-slate-900/90" />

      {/* Main animated comparison */}
      <div className="absolute inset-0 flex items-center justify-center p-8 lg:p-14">
        <div className="relative w-full max-w-2xl aspect-[4/3] rounded-2xl overflow-hidden border border-cyan-500/25 shadow-glow scan-frame">
          {SHOWCASE_SLIDES.map((s, i) => (
            <img
              key={s.src}
              src={s.src}
              alt={s.label}
              className={`absolute inset-0 w-full h-full object-cover transition-all duration-1000 ${
                i === slide ? "opacity-100 scale-100" : "opacity-0 scale-105"
              }`}
            />
          ))}

          {/* Scan wipe overlay */}
          <div
            className={`absolute inset-y-0 left-0 bg-gradient-to-r from-cyan-400/20 via-emerald-400/30 to-transparent pointer-events-none transition-all duration-[3500ms] ease-in-out ${
              scanning ? "w-[72%]" : "w-0"
            }`}
            style={{ mixBlendMode: "screen" }}
          />
          <div
            className={`scan-beam absolute inset-y-0 w-1 bg-gradient-to-b from-transparent via-cyan-300 to-transparent shadow-[0_0_24px_4px_rgba(34,211,238,0.6)] transition-all duration-[3500ms] ease-in-out ${
              scanning ? "left-[72%]" : "left-0 opacity-0"
            }`}
          />

          {/* HUD corners */}
          <div className="absolute top-3 left-3 flex gap-2">
            <span className="hud-tag">LIVE ANALYSIS</span>
            <span className="hud-tag hud-tag-emerald">IPA ENGINE</span>
          </div>
          <div className="absolute top-3 right-3 font-mono text-[10px] text-cyan-400/80 tracking-widest">
            {String(slide + 1).padStart(2, "0")} / {String(SHOWCASE_SLIDES.length).padStart(2, "0")}
          </div>
          <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent p-5 pt-16">
            <p className="font-display font-semibold text-white text-lg">{current.label}</p>
            <p className="text-sm text-cyan-300/80 mt-0.5">{current.sub}</p>
          </div>

          {/* Crosshair */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-20" viewBox="0 0 100 100" preserveAspectRatio="none">
            <line x1="50" y1="0" x2="50" y2="100" stroke="cyan" strokeWidth="0.08" />
            <line x1="0" y1="50" x2="100" y2="50" stroke="cyan" strokeWidth="0.08" />
            <circle cx="50" cy="50" r="18" fill="none" stroke="cyan" strokeWidth="0.12" strokeDasharray="1 2" className="animate-spin-slow origin-center" style={{ transformOrigin: "50px 50px" }} />
          </svg>
        </div>
      </div>

      {/* Floating satellite images */}
      <FloatingCard
        src={MICROSCOPY_IMAGES.fluorescenceRing}
        alt="Fluorescence ring"
        className="top-[8%] right-[6%] w-36 h-28 lg:w-44 lg:h-32 animate-float"
      />
      <FloatingCard
        src={MICROSCOPY_IMAGES.cellsSemBlue}
        alt="SEM cells"
        className="bottom-[12%] left-[5%] w-40 h-32 lg:w-48 lg:h-36 animate-float-delayed"
      />
      <FloatingCard
        src={MICROSCOPY_IMAGES.networkClusters}
        alt="Cell network"
        className="bottom-[8%] right-[8%] w-32 h-24 lg:w-36 lg:h-28 animate-float"
        small
      />

      {/* Slide dots */}
      <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex gap-2">
        {SHOWCASE_SLIDES.map((_, i) => (
          <button
            key={i}
            type="button"
            aria-label={`Slide ${i + 1}`}
            onClick={() => setSlide(i)}
            className={`h-1.5 rounded-full transition-all duration-300 ${
              i === slide ? "w-8 bg-cyan-400" : "w-1.5 bg-white/20 hover:bg-white/40"
            }`}
          />
        ))}
      </div>

      {/* Side stats */}
      <div className="absolute left-6 top-1/2 -translate-y-1/2 hidden xl:flex flex-col gap-4">
        {[
          { v: "99.2%", l: "Segmentation accuracy" },
          { v: "< 2s", l: "Per-field inference" },
          { v: "7", l: "AI model pipelines" },
        ].map((s) => (
          <div key={s.l} className="glass-panel px-4 py-3 rounded-lg border border-ink-line min-w-[160px] bg-black/40 backdrop-blur-md">
            <p className="font-serif text-xl font-semibold text-white">{s.v}</p>
            <p className="text-[10px] font-mono uppercase tracking-wider text-white/60 mt-0.5">{s.l}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function FloatingCard({
  src,
  alt,
  className,
  small,
}: {
  src: string;
  alt: string;
  className: string;
  delay?: string;
  small?: boolean;
}) {
  return (
    <div className={`absolute hidden md:block ${className}`}>
      <div className={`relative rounded-xl overflow-hidden border border-white/15 shadow-2xl ${small ? "opacity-70" : "opacity-90"} hover:opacity-100 transition-opacity`}>
        <img src={src} alt={alt} className="w-full h-full object-cover ken-burns" />
        <div className="absolute inset-0 bg-gradient-to-t from-black/60 to-transparent" />
        <div className="scan-beam-vertical absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-cyan-400/60 to-transparent animate-scan-vertical" />
      </div>
    </div>
  );
}

function AmbientLayer() {
  const items = [
    { src: MICROSCOPY_IMAGES.fluorescenceRing, pos: "top-[-8%] right-[-5%] w-[42vw] max-w-xl", anim: "animate-float", opacity: "opacity-[0.10]" },
    { src: MICROSCOPY_IMAGES.cellsSemBlue, pos: "bottom-[-12%] left-[-8%] w-[48vw] max-w-2xl", anim: "animate-float-delayed", opacity: "opacity-[0.09]" },
    { src: MICROSCOPY_IMAGES.networkClusters, pos: "top-[25%] left-[50%] w-[32vw] max-w-lg", anim: "animate-float", opacity: "opacity-[0.08]" },
    { src: MICROSCOPY_IMAGES.segmentationColor, pos: "bottom-[15%] right-[5%] w-[28vw] max-w-md", anim: "animate-float-delayed", opacity: "opacity-[0.07]" },
    { src: MICROSCOPY_IMAGES.comparisonGreen, pos: "top-[55%] left-[-5%] w-[30vw] max-w-md", anim: "animate-float", opacity: "opacity-[0.06]" },
    { src: MICROSCOPY_IMAGES.cellsOutline, pos: "top-[5%] left-[20%] w-[22vw] max-w-sm", anim: "animate-float-delayed", opacity: "opacity-[0.05]" },
  ];

  return (
    <div className="fixed inset-0 -z-10 overflow-hidden pointer-events-none" aria-hidden>
      {items.map((item) => (
        <div key={item.src} className={`absolute ${item.pos} ${item.anim}`}>
          <img
            src={item.src}
            alt=""
            className={`w-full rounded-2xl object-cover blur-[0.5px] ${item.opacity} ken-burns-slow`}
          />
        </div>
      ))}
      <div className="absolute inset-0 bg-gradient-to-br from-lab-900/80 via-lab-900/92 to-lab-950/95" />
    </div>
  );
}

function BannerStrip() {
  const track = [...SHOWCASE_SLIDES, ...SHOWCASE_SLIDES];

  return (
    <div className="relative overflow-hidden rounded-xl border border-ink-line bg-lab-850/60 h-28 sm:h-32">
      <div className="flex h-full gap-3 animate-marquee w-max py-3">
        {track.map((s, i) => (
          <div key={`${s.src}-${i}`} className="relative shrink-0 h-full aspect-video rounded-lg overflow-hidden group">
            <img src={s.src} alt={s.label} className="h-full w-full object-cover opacity-70 group-hover:opacity-95 transition-opacity ken-burns-slow" />
            <div className="absolute inset-0 bg-gradient-to-t from-black/70 via-transparent to-transparent" />
            <div className="scan-beam-vertical absolute left-0 right-0 h-0.5 bg-gradient-to-r from-transparent via-cyan-400/50 to-transparent animate-scan-vertical" />
            <span className="absolute bottom-2 left-2 text-[9px] font-mono uppercase tracking-wider text-cyan-300/90 bg-lab-950/80 px-2 py-0.5 rounded border border-ink-line">
              {s.label}
            </span>
          </div>
        ))}
      </div>
      <div className="absolute inset-y-0 left-0 w-20 bg-gradient-to-r from-lab-900 to-transparent pointer-events-none" />
      <div className="absolute inset-y-0 right-0 w-20 bg-gradient-to-l from-lab-900 to-transparent pointer-events-none" />
    </div>
  );
}

function CompactCarousel() {
  const [idx, setIdx] = useState(0);

  useEffect(() => {
    const t = setInterval(() => setIdx((i) => (i + 1) % SHOWCASE_SLIDES.length), 5000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="relative rounded-xl overflow-hidden aspect-[16/7] border border-ink-line bg-lab-850/80">
      {SHOWCASE_SLIDES.map((s, i) => (
        <img
          key={s.src}
          src={s.src}
          alt={s.label}
            className={`absolute inset-0 w-full h-full object-cover transition-opacity duration-700 ${
            i === idx ? "opacity-35" : "opacity-0"
          }`}
        />
      ))}
      <div className="absolute inset-0 bg-gradient-to-r from-lab-900/95 via-lab-900/80 to-lab-900/50" />
      <div className="scan-beam absolute inset-y-0 w-0.5 bg-status-ok/30 animate-scan" />
      <div className="absolute inset-0 flex flex-col justify-center px-8 max-w-lg">
        <span className="badge-ai w-fit mb-3">AI Microscopy</span>
        <h2 className="font-serif text-2xl sm:text-3xl font-semibold text-ink leading-tight">
          {SHOWCASE_SLIDES[idx].label}
        </h2>
        <p className="text-ink-muted mt-2 text-sm font-sans">{SHOWCASE_SLIDES[idx].sub}</p>
      </div>
    </div>
  );
}
