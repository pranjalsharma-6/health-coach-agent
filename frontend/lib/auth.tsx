"use client";

/**
 * Client-side auth state.
 *
 * The token lives in localStorage and every request carries it as a Bearer
 * header. That's the right trade-off for a portfolio app: no cookie/CSRF
 * machinery, and the API stays stateless and callable from anywhere.
 */

import { useRouter } from "next/navigation";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { ApiError, api, clearToken, getToken, setToken } from "./api";
import type { User } from "./types";

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string) => Promise<User>;
  register: (
    email: string,
    password: string,
    fullName: string,
  ) => Promise<User>;
  logout: () => void;
  refresh: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const router = useRouter();

  // Restore the session on first mount.
  useEffect(() => {
    let cancelled = false;

    async function restore() {
      if (!getToken()) {
        setLoading(false);
        return;
      }
      try {
        const me = await api.auth.me();
        if (!cancelled) setUser(me);
      } catch (error) {
        // An expired or tampered token. Drop it silently.
        if (error instanceof ApiError && error.isAuthError) clearToken();
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    void restore();
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const result = await api.auth.login(email, password);
    setToken(result.access_token);
    setUser(result.user);
    return result.user;
  }, []);

  const register = useCallback(
    async (email: string, password: string, fullName: string) => {
      const result = await api.auth.register(email, password, fullName);
      setToken(result.access_token);
      setUser(result.user);
      return result.user;
    },
    [],
  );

  const logout = useCallback(() => {
    clearToken();
    setUser(null);
    router.push("/");
  }, [router]);

  const refresh = useCallback(async () => {
    if (!getToken()) return;
    try {
      setUser(await api.auth.me());
    } catch {
      /* leave the existing user in place on a transient failure */
    }
  }, []);

  const value = useMemo(
    () => ({ user, loading, login, register, logout, refresh }),
    [user, loading, login, register, logout, refresh],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used inside an <AuthProvider>.");
  }
  return context;
}

/**
 * Redirect to sign-in if there's no session, or to onboarding if the profile
 * isn't set up yet. Returns the loading state so pages can render a skeleton.
 */
export function useRequireAuth(options: { requireOnboarded?: boolean } = {}) {
  const { user, loading } = useAuth();
  const router = useRouter();
  const { requireOnboarded = true } = options;

  useEffect(() => {
    if (loading) return;
    if (!user) {
      router.replace("/login");
      return;
    }
    if (requireOnboarded && !user.onboarded) {
      router.replace("/onboarding");
    }
  }, [user, loading, requireOnboarded, router]);

  return { user, loading };
}
