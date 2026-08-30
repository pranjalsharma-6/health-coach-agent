"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect } from "react";

import { Button } from "@/components/ui";
import { useAuth } from "@/lib/auth";

export default function LandingPage() {
  const { user, loading } = useAuth();
  const router = useRouter();

  // Signed-in visitors don't need the pitch.
  useEffect(() => {
    if (!loading && user) {
      router.replace(user.onboarded ? "/dashboard" : "/onboarding");
    }
  }, [user, loading, router]);

  return (
    <div className="min-h-screen">
      <header className="max-w-6xl mx-auto px-6 py-6 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Logo />
          <span className="font-display text-xl font-semibold tracking-tight">
            Kaya
          </span>
        </div>
        <nav className="flex items-center gap-2">
          <Link href="/login">
            <Button variant="ghost" size="sm">
              Sign in
            </Button>
          </Link>
          <Link href="/register">
            <Button size="sm">Get started</Button>
          </Link>
        </nav>
      </header>

      <main>
        <section className="max-w-6xl mx-auto px-6 pt-16 pb-24 text-center">
          <p className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-brand-50 dark:bg-brand-950 text-brand-700 dark:text-brand-300 text-xs font-medium mb-6 animate-fade-up">
            <span className="size-1.5 rounded-full bg-brand-500 animate-pulse-ring" />
            Autonomous agent · LangGraph
          </p>

          <h1
            className="font-display text-4xl sm:text-6xl font-semibold tracking-tight text-ink max-w-4xl mx-auto text-balance animate-fade-up"
            style={{ animationDelay: "60ms" }}
          >
            Every nutrition app is a{" "}
            <span className="text-ink-muted line-through decoration-clay-400 decoration-2">
              logging
            </span>{" "}
            app.
            <br />
            Kaya is a <span className="text-brand-600">planning</span> agent.
          </h1>

          <p
            className="mt-6 text-lg text-ink-soft max-w-2xl mx-auto text-pretty animate-fade-up"
            style={{ animationDelay: "120ms" }}
          >
            Other apps tell you that you ate 2,400 calories. They don&apos;t change
            your dinner because you skipped lunch. Kaya notices, works out why it
            matters for your goal, and rewrites the rest of your week, without
            you asking.
          </p>

          <div
            className="mt-9 flex flex-col sm:flex-row gap-3 justify-center animate-fade-up"
            style={{ animationDelay: "180ms" }}
          >
            <Link href="/register">
              <Button size="lg" className="w-full sm:w-auto">
                Build my plan
              </Button>
            </Link>
            <Link href="/login">
              <Button variant="secondary" size="lg" className="w-full sm:w-auto">
                I already have an account
              </Button>
            </Link>
          </div>
        </section>

        <section className="max-w-6xl mx-auto px-6 pb-24">
          <div className="grid md:grid-cols-3 gap-5">
            <FeatureCard
              emoji="🔁"
              title="It adapts on its own"
              body="Skip lunch and the remaining meals absorb the difference. Skip breakfast three days running and it stops planning breakfast, because the plan is what's wrong, not you."
            />
            <FeatureCard
              emoji="🙏"
              title="It knows how you actually eat"
              body="Vegetarian, eggetarian, vegan, Jain and halal, modelled properly rather than as one checkbox. No quinoa bowls for someone who eats dal-chawal."
            />
            <FeatureCard
              emoji="🧮"
              title="It doesn't invent your numbers"
              body="Calories and macros come from validated equations with safety floors. The model picks the food; it never picks the maths."
            />
          </div>
        </section>

        <section className="max-w-4xl mx-auto px-6 pb-28">
          <h2 className="font-display text-2xl font-semibold text-center mb-10">
            How the loop works
          </h2>
          <ol className="grid sm:grid-cols-4 gap-4">
            {[
              { n: "1", t: "Sense", d: "You log what you actually ate." },
              { n: "2", t: "Evaluate", d: "It measures the gap against your targets." },
              { n: "3", t: "Decide", d: "Rebalance the day, or rebuild the plan." },
              { n: "4", t: "Act", d: "It rewrites your meals and tells you why." },
            ].map((step) => (
              <li key={step.n} className="text-center">
                <div className="mx-auto size-9 rounded-full bg-brand-600 text-white grid place-items-center font-semibold text-sm mb-3">
                  {step.n}
                </div>
                <p className="font-medium text-ink">{step.t}</p>
                <p className="text-sm text-ink-soft mt-1">{step.d}</p>
              </li>
            ))}
          </ol>
        </section>
      </main>

      <footer className="border-t border-line">
        <div className="max-w-6xl mx-auto px-6 py-8 text-sm text-ink-muted flex flex-col sm:flex-row gap-2 justify-between">
          <p>Kaya. काया, &ldquo;body&rdquo; in Sanskrit and Hindi.</p>
          <p>Built with FastAPI, LangGraph and Next.js.</p>
        </div>
      </footer>
    </div>
  );
}

function Logo() {
  return (
    <div className="size-8 rounded-lg bg-brand-600 grid place-items-center text-white font-display font-bold">
      K
    </div>
  );
}

function FeatureCard({
  emoji,
  title,
  body,
}: {
  emoji: string;
  title: string;
  body: string;
}) {
  return (
    <div className="bg-card border border-line rounded-2xl p-6">
      <div className="text-2xl mb-3">{emoji}</div>
      <h3 className="font-display text-lg font-semibold text-ink mb-2">
        {title}
      </h3>
      <p className="text-sm text-ink-soft leading-relaxed">{body}</p>
    </div>
  );
}
