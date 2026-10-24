import { Link } from "react-router-dom";

export default function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <Link to="/analyze" className="flex items-center gap-3 group">
      <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-emerald-400 to-cyan-600 flex items-center justify-center shadow-lg shadow-cyan-500/20">
        <svg viewBox="0 0 24 24" className="w-6 h-6 text-white" fill="none" stroke="currentColor" strokeWidth="1.5">
          <circle cx="12" cy="12" r="3" fill="currentColor" fillOpacity="0.3" />
          <circle cx="7" cy="9" r="1.5" />
          <circle cx="17" cy="9" r="1.5" />
          <circle cx="8" cy="15" r="1.2" />
          <circle cx="16" cy="15" r="1.2" />
        </svg>
      </div>
      {!compact && (
        <div className="hidden sm:block">
          <p className="font-serif font-semibold text-base leading-tight text-ink">IPA Platform</p>
          <p className="text-[10px] font-mono uppercase tracking-[0.18em] text-ink-faint">Microscopy AI</p>
        </div>
      )}
    </Link>
  );
}
