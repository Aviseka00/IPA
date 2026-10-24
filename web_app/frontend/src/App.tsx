import { Navigate, Route, Routes } from "react-router-dom";
import { AuthProvider, useAuth } from "./AuthContext";
import BioBackground from "./components/BioBackground";
import Layout from "./Layout";
import Analyze from "./pages/Analyze";
import Compliance from "./pages/Compliance";
import Dashboard from "./pages/Dashboard";
import Login from "./pages/Login";
import Register from "./pages/Register";
import ForgotPassword from "./pages/ForgotPassword";

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const { employeeId, loading } = useAuth();
  if (loading) {
    return (
      <div className="min-h-screen flex flex-col items-center justify-center relative">
        <BioBackground />
        <div className="loading-ring" />
        <p className="mt-6 text-sm font-mono text-ink-faint tracking-widest uppercase">
          Initializing session
        </p>
      </div>
    );
  }
  if (!employeeId) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route path="/register" element={<Register />} />
        <Route path="/forgot-password" element={<ForgotPassword />} />
        <Route
          element={
            <PrivateRoute>
              <Layout />
            </PrivateRoute>
          }
        >
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/compliance" element={<Compliance />} />
        </Route>
        <Route path="*" element={<Navigate to="/analyze" replace />} />
      </Routes>
    </AuthProvider>
  );
}
