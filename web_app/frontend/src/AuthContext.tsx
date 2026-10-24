import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";
import { api } from "./api";

type AuthState = {
  employeeId: string | null;
  loading: boolean;
  login: (id: string, password: string) => Promise<void>;
  register: (id: string, password: string) => Promise<void>;
  logout: () => void;
};

const AuthContext = createContext<AuthState | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [employeeId, setEmployeeId] = useState<string | null>(
    () => localStorage.getItem("ipa_employee_id")
  );
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Whenever the IPA UI opens, ensure Ani is running alongside it
    void api.ensureAniRunning();

    const token = localStorage.getItem("ipa_token");
    if (!token) {
      setLoading(false);
      return;
    }
    api
      .me()
      .then((u) => setEmployeeId(u.employee_id))
      .catch(() => {
        localStorage.removeItem("ipa_token");
        localStorage.removeItem("ipa_employee_id");
        setEmployeeId(null);
      })
      .finally(() => setLoading(false));
  }, []);

  const login = useCallback(async (id: string, password: string) => {
    const res = await api.login(id, password);
    localStorage.setItem("ipa_token", res.access_token);
    localStorage.setItem("ipa_employee_id", res.employee_id);
    setEmployeeId(res.employee_id);
  }, []);

  const register = useCallback(async (id: string, password: string) => {
    await api.register(id, password);
    await login(id, password);
  }, [login]);

  const logout = useCallback(() => {
    localStorage.removeItem("ipa_token");
    localStorage.removeItem("ipa_employee_id");
    setEmployeeId(null);
  }, []);

  return (
    <AuthContext.Provider value={{ employeeId, loading, login, register, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth outside provider");
  return ctx;
}
