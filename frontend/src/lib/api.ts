// Typed client for the VoicePrint FastAPI backend.
// All requests target ${API_URL}/api/v1/... and mirror the schemas in
// backend/voiceprint/schemas/profile.py exactly.

export const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const BASE = `${API_URL}/api/v1`;

// The six normalized fingerprint axes, in the canonical order the backend
// returns them (see backend/voiceprint/services/feedback.py RADAR_AXES).
export const AXIS_KEYS = [
  "avg_sentence_length",
  "type_token_ratio",
  "contraction_rate",
  "passive_voice_ratio",
  "function_word_ratio",
  "transition_word_rate",
] as const;

export type AxisKey = (typeof AXIS_KEYS)[number];

export const AXIS_LABELS: Record<AxisKey, string> = {
  avg_sentence_length: "Sentence length",
  type_token_ratio: "Lexical diversity",
  contraction_rate: "Contractions",
  passive_voice_ratio: "Passive voice",
  function_word_ratio: "Function words",
  transition_word_rate: "Transitions",
};

export type Axes = Partial<Record<AxisKey, number>>;

// Mirrors StyleProfileCreate.
export interface StyleProfileCreate {
  user_id: string;
  samples: string[];
  profile_name?: string | null;
}

// Mirrors StyleProfileResponse.
export interface StyleProfileResponse {
  profile_id: string;
  user_id: string;
  profile_name: string | null;
  created_at: string;
  updated_at: string;
  total_words: number;
  num_samples: number;
  linguistic_features: Record<string, unknown>;
  stylometric_features: Record<string, unknown>;
  embedding_metadata: Record<string, unknown>;
}

// Mirrors AdaptRequest.
export interface AdaptRequest {
  profile_id: string;
  source_draft: string;
  use_llm?: boolean;
  api_key?: string | null;
  base_url?: string | null;
  model?: string | null;
}

export type DiffLabel = "improved" | "regressed" | "same";

export interface SentenceDiffItem {
  sentence: string;
  voice_match: number;
  label: DiffLabel;
}

export interface RadarFeedback {
  axes: string[];
  author: Axes;
  before: Axes;
  after: Axes;
}

export interface AdaptFeedback {
  radar: RadarFeedback;
  axis_delta: Record<string, number>;
  sentence_diff: SentenceDiffItem[];
}

// Mirrors AdaptResponse.
export interface AdaptResponse {
  source_draft: string;
  adapted_text: string;
  voice_match_before: number;
  voice_match_after: number;
  rewrite_path: string;
  validation: Record<string, unknown> & {
    semantic_preservation?: number;
  };
  feedback: AdaptFeedback;
}

export interface HealthResponse {
  status: string;
  service: string;
}

// A friendly error type so the UI can distinguish "backend offline" from a
// validation error the backend returned.
export class ApiError extends Error {
  readonly status: number;
  readonly offline: boolean;
  constructor(message: string, status: number, offline = false) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.offline = offline;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        ...(init?.headers ?? {}),
      },
    });
  } catch {
    throw new ApiError(
      "Could not reach the VoicePrint backend. Make sure it is running.",
      0,
      true,
    );
  }

  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try {
      const body = await res.json();
      if (typeof body?.detail === "string") {
        detail = body.detail;
      } else if (Array.isArray(body?.detail) && body.detail[0]?.msg) {
        detail = body.detail[0].msg;
      }
    } catch {
      // Keep the default detail message.
    }
    throw new ApiError(detail, res.status);
  }

  if (res.status === 204) {
    return undefined as T;
  }
  return (await res.json()) as T;
}

export function createProfile(
  body: StyleProfileCreate,
): Promise<StyleProfileResponse> {
  return request<StyleProfileResponse>("/profiles", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function getProfile(
  profileId: string,
): Promise<StyleProfileResponse> {
  return request<StyleProfileResponse>(
    `/profiles/${encodeURIComponent(profileId)}`,
  );
}

export function adaptDraft(body: AdaptRequest): Promise<AdaptResponse> {
  return request<AdaptResponse>("/adapt", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function checkHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}
