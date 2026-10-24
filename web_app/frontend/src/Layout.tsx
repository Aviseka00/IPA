import { Link, Outlet, useLocation } from "react-router-dom";
import { useAuth } from "./AuthContext";
import BioBackground from "./components/BioBackground";
import Logo from "./components/Logo";
import { IconAnalyze, IconHistory, IconLogout } from "./components/Icons";

export default function Layout() {
  const { employeeId, logout } = useAuth();
  const loc = useLocation();

  const nav = [
    { to: "/analyze", label: "Analyze", icon: IconAnalyze, desc: "Run inference" },
    { to: "/dashboard", label: "History", icon: IconHistory, desc: "Open a batch and review" },
    { to: "/compliance", label: "Part 11", icon: IconHistory, desc: "21 CFR Part 11 / GAMP 5" },
  ];

  return (
    <div className="min-h-screen flex flex-col relative">
      <BioBackground />

      <header className="glass-panel border-b border-ink-line sticky top-0 z-50 bg-lab-900/95">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 h-[4.25rem] flex items-center justify-between gap-4">
          <div className="flex items-center gap-6 lg:gap-10">
            <Logo />
            <nav className="flex gap-1">
              {nav.map((n) => {
                const active = loc.pathname.startsWith(n.to);
                const Icon = n.icon;
                return (
                  <Link
                    key={n.to}
                    to={n.to}
                    className={active ? "nav-link-active" : "nav-link-inactive"}
                    title={n.desc}
                  >
                    <Icon className="w-4 h-4 shrink-0" />
                    <span>{n.label}</span>
                  </Link>
                );
              })}
            </nav>
          </div>

          <div className="flex items-center gap-3 sm:gap-5">
            <div className="hidden md:flex items-center gap-2 badge-live">
              <span className="w-1.5 h-1.5 rounded-full bg-status-ok animate-pulse" />
              Engine online
            </div>
            <div className="h-8 w-px bg-ink-line hidden sm:block" />
            <div className="text-right hidden sm:block">
              <p className="section-label">Operator</p>
              <p className="text-sm font-mono text-ink">{employeeId}</p>
            </div>
            <button
              type="button"
              onClick={logout}
              className="btn-ghost flex items-center gap-2 text-sm py-2"
              title="Sign out"
            >
              <IconLogout className="w-4 h-4" />
              <span className="hidden sm:inline">Sign out</span>
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-8 sm:py-10">
        <Outlet />
      </main>

      <footer className="border-t border-ink-line py-4 mt-auto bg-lab-900/80">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 flex flex-wrap items-center justify-between gap-2 text-xs text-ink-faint font-mono">
          <span>IPA Platform · 21 CFR Part 11 controls · GAMP 5 CSV package</span>
          <span className="italic font-sans text-ink-muted">AI analysis · Human review · E-signature</span>
        </div>
      </footer>
    </div>
  );
}
