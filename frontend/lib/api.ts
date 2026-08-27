/**
 * Typed client for the Kaya API.
 *
 * One place that knows about base URLs, auth headers and error shapes, so no
 * component ever calls `fetch` directly.
 */

import type {
  AdherenceSnapshot,
  AgentEvent,
  AgentStep,
  AuthResponse,
  DailyLog,
  MealLogResponse,
  MealStatus,
  Plan,
  PlanSummary,
  Profile,
  ProfileDraft,
  MacroCheck,
  Recipe,
  TargetsResponse,
  WeightPoint,
} from "./types";

interface RecipeResponse {
  meal_id: string;
  meal_name: string;
  recipe: Recipe;
  cached: boolean;
  macro_check: MacroCheck | null;
}

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";

const TOKEN_KEY = "kaya_token";

// --------------------------------------------------------------------------
// Token storage
// --------------------------------------------------------------------------

export function getToken(): string | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage.getItem(TOKEN_KEY);
  } catch {
    // Private mode or blocked site data — treat as signed out rather than crash.
    return null;
  }
}

export function setToken(token: string): void {
  try {
    window.localStorage.setItem(TOKEN_KEY, token);
  } catch {
    /* non-fatal: the session just won't survive a reload */
  }
}

export function clearToken(): void {
  try {
    window.localStorage.removeItem(TOKEN_KEY);
  } catch {
    /* ignore */
  }
}

// --------------------------------------------------------------------------
// Core request helper
// --------------------------------------------------------------------------

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }

  /** True when the user needs to complete onboarding before this call works. */
  get needsOnboarding(): boolean {
    return this.status === 409;
  }

  get isAuthError(): boolean {
    return this.status === 401;
  }
}

async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const token = getToken();

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    ...((options.headers as Record<string, string>) ?? {}),
  };
  if (token) headers.Authorization = `Bearer ${token}`;

  let response: Response;
  try {
    response = await fetch(`${API_BASE}${path}`, { ...options, headers });
  } catch {
    throw new ApiError(
      0,
      "Can't reach the server. Is the backend running on port 8000?",
    );
  }

  if (response.status === 204) return undefined as T;

  const text = await response.text();
  const body = text ? safeParse(text) : null;

  if (!response.ok) {
    throw new ApiError(response.status, extractDetail(body, response.status));
  }

  return body as T;
}

function safeParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

/** FastAPI returns `detail` as a string, or as a list for validation errors. */
function extractDetail(body: unknown, status: number): string {
  if (typeof body === "string" && body) return body;

  if (body && typeof body === "object" && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          if (item && typeof item === "object" && "msg" in item) {
            const loc = Array.isArray((item as { loc?: unknown[] }).loc)
              ? (item as { loc: unknown[] }).loc.slice(1).join(".")
              : "";
            const msg = String((item as { msg: unknown }).msg);
            return loc ? `${loc}: ${msg}` : msg;
          }
          return String(item);
        })
        .join("; ");
    }
  }

  return `Request failed (${status}).`;
}

// --------------------------------------------------------------------------
// Endpoints
// --------------------------------------------------------------------------

export const api = {
  auth: {
    register: (email: string, password: string, fullName: string) =>
      request<AuthResponse>("/auth/register", {
        method: "POST",
        body: JSON.stringify({ email, password, full_name: fullName }),
      }),

    login: (email: string, password: string) =>
      request<AuthResponse>("/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      }),

    me: () => request<AuthResponse["user"]>("/auth/me"),
  },

  profile: {
    create: (draft: ProfileDraft) =>
      request<Profile>("/profile", {
        method: "POST",
        body: JSON.stringify(draft),
      }),

    get: () => request<Profile>("/profile"),

    update: (changes: Partial<ProfileDraft>) =>
      request<Profile>("/profile", {
        method: "PATCH",
        body: JSON.stringify(changes),
      }),

    targets: () => request<TargetsResponse>("/profile/targets"),
  },

  plans: {
    active: () => request<Plan | null>("/plans/active"),
    history: () => request<PlanSummary[]>("/plans/history"),
    byId: (id: string) => request<Plan>(`/plans/${id}`),
    recipe: (mealId: string) =>
      request<RecipeResponse>(
        `/plans/meals/${encodeURIComponent(mealId)}/recipe`,
        { method: "POST" },
      ),
  },

  logs: {
    logMeal: (payload: {
      meal_id: string;
      status: MealStatus;
      actual_calories_kcal?: number | null;
      actual_protein_g?: number | null;
      substitute_name?: string | null;
      note?: string | null;
    }) =>
      request<MealLogResponse>("/logs/meals", {
        method: "POST",
        body: JSON.stringify(payload),
      }),

    logMetrics: (payload: {
      weight_kg?: number;
      steps?: number;
      sleep_hours?: number;
      water_ml?: number;
    }) =>
      request<DailyLog>("/logs/metrics", {
        method: "POST",
        body: JSON.stringify(payload),
      }),

    today: () => request<DailyLog>("/logs/today"),
    adherence: () => request<AdherenceSnapshot>("/logs/adherence"),
    weight: (days = 90) => request<WeightPoint[]>(`/logs/weight?days=${days}`),
  },

  agent: {
    events: () => request<AgentEvent[]>("/agent/events"),

    /**
     * Run the agent, yielding each step as it arrives.
     *
     * Uses fetch + a manual SSE parse rather than EventSource, because
     * EventSource can't send an Authorization header.
     */
    async *stream(forceReplan = false): AsyncGenerator<AgentStep> {
      const token = getToken();
      const response = await fetch(
        `${API_BASE}/agent/stream?force_replan=${forceReplan}`,
        {
          headers: token ? { Authorization: `Bearer ${token}` } : {},
        },
      );

      if (!response.ok) {
        const text = await response.text();
        throw new ApiError(response.status, extractDetail(safeParse(text), response.status));
      }
      if (!response.body) {
        throw new ApiError(0, "The server returned an empty stream.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line.
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          const line = frame
            .split("\n")
            .find((l) => l.startsWith("data:"));
          if (!line) continue;

          try {
            yield JSON.parse(line.slice(5).trim()) as AgentStep;
          } catch {
            // A partial frame — skip it rather than killing the run.
          }
        }
      }
    },
  },

  health: () =>
    fetch(`${API_BASE.replace("/api/v1", "")}/health`).then((r) => r.json()),
};
