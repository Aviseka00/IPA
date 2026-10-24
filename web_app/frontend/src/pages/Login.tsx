import { FormEvent, useState } from "react";
import { Link, useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "../AuthContext";
import BioBackground from "../components/BioBackground";
import Logo from "../components/Logo";
import MicroscopyShowcase from "../components/MicroscopyShowcase";

export default function Login() {
  const { login } = useAuth();
  const nav = useNavigate();
  const [search] = useSearchParams();
  const sessionMsg = search.get("reason");
  const [employeeId, setEmployeeId] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    setBusy(true);
    try {
      await login(employeeId.trim(), password.trim());
      nav("/analyze");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="min-h-screen flex relative bg-lab-900">
      <BioBackground variant="auth" />

      <div className="relative z-10 w-full lg:w-[42%] xl:w-[38%] flex flex-col min-h-screen">
        <div className="flex-1 flex items-center justify-center p-6 sm:p-10 lg:p-12">
          <div className="w-full max-w-md">
            <div className="mb-8 lg:hidden flex justify-center">
              <Logo />
            </div>

            <div className="glass-panel rounded-xl p-8 sm:p-10 border border-ink-line shadow-soft">
              <div className="hidden lg:block mb-8">
                <Logo />
              </div>

              <div className="mb-8">
                <p className="section-label mb-2">Secure access</p>
                <h1 className="font-serif text-3xl font-semibold text-ink">Welcome back</h1>
                <p className="text-ink-muted mt-2 text-sm leading-relaxed font-sans">
                  Sign in to run AI-powered infected cell detection on microscopy slides.
                </p>
              </div>

              {sessionMsg && (
                <p className="text-status-warn text-sm badge-warn rounded-lg px-4 py-3 mb-5 w-full justify-start">
                  {sessionMsg}
                </p>
              )}

              <form onSubmit={onSubmit} className="space-y-5">
                <div>
                  <label className="block section-label mb-2">Employee ID</label>
                  <input
                    className="input-field font-mono text-sm"
                    value={employeeId}
                    onChange={(e) => setEmployeeId(e.target.value)}
                    required
                    autoComplete="username"
                    placeholder="e.g. 80100173"
                  />
                </div>
                <div>
                  <label className="block section-label mb-2">Password</label>
                  <input
                    type="password"
                    className="input-field"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    autoComplete="current-password"
                    placeholder="••••••••"
                  />
                </div>
                {error && (
                  <p className="text-status-warn text-sm badge-warn rounded-lg px-4 py-3 w-full justify-start">
                    {error}
                  </p>
                )}
                <button type="submit" className="btn-primary w-full" disabled={busy}>
                  {busy ? "Authenticating…" : "Sign in to platform"}
                </button>
              </form>

              <p className="text-center text-sm text-ink-muted mt-6 font-sans">
                <Link to="/forgot-password" className="text-cyan-400 hover:underline font-medium">
                  Forgot password?
                </Link>
              </p>

              <p className="text-center text-sm text-ink-muted mt-4 font-sans">
                New operator?{" "}
                <Link to="/register" className="text-cyan-400 hover:underline font-medium">
                  Create account
                </Link>
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="hidden lg:block lg:flex-1 relative min-h-screen border-l border-ink-line">
        <MicroscopyShowcase variant="hero" />
      </div>
    </div>
  );
}
