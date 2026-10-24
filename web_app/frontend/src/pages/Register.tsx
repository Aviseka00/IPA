import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../AuthContext";
import BioBackground from "../components/BioBackground";
import Logo from "../components/Logo";
import MicroscopyShowcase from "../components/MicroscopyShowcase";

export default function Register() {
  const { register } = useAuth();
  const nav = useNavigate();
  const [employeeId, setEmployeeId] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (password !== confirm) {
      setError("Passwords do not match");
      return;
    }
    setError("");
    setBusy(true);
    try {
      await register(employeeId.trim(), password);
      nav("/analyze");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed");
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
                <span className="badge-live mb-3">New operator</span>
                <h1 className="font-serif text-3xl font-semibold text-ink mt-2">Create account</h1>
                <p className="text-ink-muted mt-2 text-sm font-sans">
                  Register with your employee ID to access the microscopy AI workbench.
                </p>
              </div>

              <form onSubmit={onSubmit} className="space-y-5">
                <div>
                  <label className="block section-label mb-2">Employee ID</label>
                  <input
                    className="input-field font-mono text-sm"
                    value={employeeId}
                    onChange={(e) => setEmployeeId(e.target.value)}
                    required
                    pattern="[A-Za-z0-9_-]+"
                    title="Letters, numbers, underscore, hyphen"
                    placeholder="Alphanumeric ID"
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
                    minLength={8}
                    placeholder="Min. 8 chars · upper, lower, digit, special"
                  />
                </div>
                <div>
                  <label className="block section-label mb-2">Confirm password</label>
                  <input
                    type="password"
                    className="input-field"
                    value={confirm}
                    onChange={(e) => setConfirm(e.target.value)}
                    required
                    placeholder="Repeat password"
                  />
                </div>
                {error && (
                  <p className="text-status-warn text-sm badge-warn rounded-lg px-4 py-3 w-full justify-start">
                    {error}
                  </p>
                )}
                <button type="submit" className="btn-primary w-full" disabled={busy}>
                  {busy ? "Creating account…" : "Register & continue"}
                </button>
              </form>

              <p className="text-center text-sm text-ink-muted mt-8 font-sans">
                <Link to="/login" className="text-cyan-400 hover:underline font-medium">
                  ← Back to sign in
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
