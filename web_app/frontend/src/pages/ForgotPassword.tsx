import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api";
import BioBackground from "../components/BioBackground";
import Logo from "../components/Logo";
import MicroscopyShowcase from "../components/MicroscopyShowcase";

export default function ForgotPassword() {
  const nav = useNavigate();
  const [employeeId, setEmployeeId] = useState("");
  const [resetCode, setResetCode] = useState("");
  const [password, setPassword] = useState("");
  const [confirm, setConfirm] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [busy, setBusy] = useState(false);

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (password !== confirm) {
      setError("Passwords do not match");
      return;
    }
    setError("");
    setSuccess("");
    setBusy(true);
    try {
      await api.resetPassword(employeeId.trim(), resetCode.trim(), password);
      setSuccess("Password updated. Redirecting to sign in…");
      setTimeout(() => nav("/login"), 1800);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Password reset failed");
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
                <p className="section-label mb-2">Account recovery</p>
                <h1 className="font-serif text-3xl font-semibold text-ink">Reset password</h1>
                <p className="text-ink-muted mt-2 text-sm leading-relaxed font-sans">
                  Passwords cannot be retrieved — set a new one using your Employee ID and the
                  reset code from your administrator.
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
                    autoComplete="username"
                    placeholder="e.g. 80100173"
                  />
                </div>
                <div>
                  <label className="block section-label mb-2">Reset code</label>
                  <input
                    className="input-field font-mono text-sm"
                    value={resetCode}
                    onChange={(e) => setResetCode(e.target.value)}
                    required
                    autoComplete="off"
                    placeholder="From administrator"
                  />
                </div>
                <div>
                  <label className="block section-label mb-2">New password</label>
                  <input
                    type="password"
                    className="input-field"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    minLength={8}
                    autoComplete="new-password"
                    placeholder="Min. 8 · upper, lower, digit, special"
                  />
                </div>
                <div>
                  <label className="block section-label mb-2">Confirm new password</label>
                  <input
                    type="password"
                    className="input-field"
                    value={confirm}
                    onChange={(e) => setConfirm(e.target.value)}
                    required
                    minLength={8}
                    autoComplete="new-password"
                    placeholder="Repeat new password"
                  />
                </div>

                {error && (
                  <p className="text-status-warn text-sm badge-warn rounded-lg px-4 py-3 w-full justify-start">
                    {error}
                  </p>
                )}
                {success && (
                  <p className="text-status-ok text-sm badge-ok rounded-lg px-4 py-3 w-full justify-start">
                    {success}
                  </p>
                )}

                <button type="submit" className="btn-primary w-full" disabled={busy}>
                  {busy ? "Updating…" : "Set new password"}
                </button>
              </form>

              <p className="text-center text-sm text-ink-muted mt-8 font-sans">
                Remember your password?{" "}
                <Link to="/login" className="text-cyan-400 hover:underline font-medium">
                  Sign in
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
