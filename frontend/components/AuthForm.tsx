"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState, type FormEvent } from "react";

import { Alert, Button, Field, Input } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { useAuth } from "@/lib/auth";

export function AuthForm({ mode }: { mode: "login" | "register" }) {
  const isRegister = mode === "register";
  const { login, register } = useAuth();
  const router = useRouter();

  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setError(null);

    if (isRegister && password.length < 8) {
      setError("Password needs to be at least 8 characters.");
      return;
    }

    setSubmitting(true);
    try {
      const user = isRegister
        ? await register(email, password, fullName)
        : await login(email, password);

      router.push(user.onboarded ? "/dashboard" : "/onboarding");
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Something went wrong. Please try again.",
      );
      setSubmitting(false);
    }
  }

  return (
    <div className="min-h-screen grid place-items-center px-6 py-12">
      <div className="w-full max-w-sm animate-fade-up">
        <Link href="/" className="flex items-center gap-2 mb-8 w-fit mx-auto">
          <div className="size-8 rounded-lg bg-brand-600 grid place-items-center text-white font-display font-bold">
            K
          </div>
          <span className="font-display text-xl font-semibold">Kaya</span>
        </Link>

        <h1 className="font-display text-2xl font-semibold text-center text-ink">
          {isRegister ? "Let's build your plan" : "Welcome back"}
        </h1>
        <p className="text-sm text-ink-soft text-center mt-1.5 mb-8">
          {isRegister
            ? "Two minutes of setup, then Kaya takes it from there."
            : "Sign in to pick up where you left off."}
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          {error && <Alert tone="error">{error}</Alert>}

          {isRegister && (
            <Field label="Your name" htmlFor="fullName">
              <Input
                id="fullName"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                placeholder="Pranjal Sharma"
                autoComplete="name"
                required
              />
            </Field>
          )}

          <Field label="Email" htmlFor="email">
            <Input
              id="email"
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              autoComplete="email"
              required
            />
          </Field>

          <Field
            label="Password"
            htmlFor="password"
            hint={isRegister ? "At least 8 characters." : undefined}
          >
            <Input
              id="password"
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              autoComplete={isRegister ? "new-password" : "current-password"}
              required
            />
          </Field>

          <Button type="submit" size="lg" loading={submitting} className="w-full">
            {isRegister ? "Create account" : "Sign in"}
          </Button>
        </form>

        <p className="text-sm text-ink-soft text-center mt-6">
          {isRegister ? "Already have an account? " : "New here? "}
          <Link
            href={isRegister ? "/login" : "/register"}
            className="text-brand-600 font-medium hover:underline"
          >
            {isRegister ? "Sign in" : "Create one"}
          </Link>
        </p>
      </div>
    </div>
  );
}
