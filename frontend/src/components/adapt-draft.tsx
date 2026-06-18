"use client";

import { useState } from "react";
import {
  adaptDraft,
  ApiError,
  type AdaptResponse,
  type DiffLabel,
  type StyleProfileResponse,
} from "@/lib/api";
import { RadarChart } from "@/components/radar-chart";
import {
  Button,
  Card,
  ErrorNote,
  Input,
  Label,
  Textarea,
} from "@/components/ui";

interface AdaptDraftProps {
  profile: StyleProfileResponse | null;
}

const DIFF_STYLES: Record<DiffLabel, string> = {
  improved:
    "bg-[var(--accent)]/15 text-[var(--foreground)] border-[var(--accent)]/30",
  regressed:
    "bg-[var(--destructive)]/15 text-[var(--foreground)] border-[var(--destructive)]/30",
  same: "bg-[var(--muted)] text-[var(--muted-foreground)] border-[var(--border)]",
};

const DIFF_LABEL_TEXT: Record<DiffLabel, string> = {
  improved: "Improved",
  regressed: "Regressed",
  same: "Unchanged",
};

function pct(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function AdaptDraft({ profile }: AdaptDraftProps) {
  const [draft, setDraft] = useState("");
  const [showKeyPanel, setShowKeyPanel] = useState(false);
  const [apiKey, setApiKey] = useState("");
  const [baseUrl, setBaseUrl] = useState("");
  const [model, setModel] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<AdaptResponse | null>(null);

  async function handleAdapt() {
    setError(null);

    if (!profile) {
      setError("Build your profile first.");
      return;
    }
    if (draft.trim().length < 10) {
      setError("Please paste a draft of at least 10 characters.");
      return;
    }

    setLoading(true);
    try {
      const res = await adaptDraft({
        profile_id: profile.profile_id,
        source_draft: draft,
        use_llm: true,
        api_key: apiKey.trim() || null,
        base_url: baseUrl.trim() || null,
        model: model.trim() || null,
      });
      setResult(res);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Something went wrong while adapting your draft.",
      );
    } finally {
      setLoading(false);
    }
  }

  const semantic =
    result && typeof result.validation.semantic_preservation === "number"
      ? result.validation.semantic_preservation
      : null;

  const pathNote = result
    ? result.rewrite_path === "llm"
      ? "Rewritten with the LLM rewriter."
      : "No API key provided, so this used the rule-based rewriter."
    : null;

  return (
    <section id="adapt" className="scroll-mt-20">
      <div className="mb-8">
        <p className="font-mono text-xs tracking-wide text-[var(--accent)]">
          02 / Adapt a draft
        </p>
        <h2 className="mt-2 text-2xl font-bold tracking-tight">
          Nudge a draft toward your voice
        </h2>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[var(--muted-foreground)]">
          Paste one of your own drafts to adapt it toward your measured voice. A
          key is optional: without one, a rule-based rewriter runs.
        </p>
      </div>

      {!profile && (
        <p className="mb-6 rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--muted)] px-3.5 py-2.5 text-sm text-[var(--muted-foreground)]">
          Build a profile in step 01 first, then your draft can be scored
          against it.
        </p>
      )}

      <div className="grid gap-8 lg:grid-cols-2">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="draft">Your draft</Label>
            <Textarea
              id="draft"
              rows={9}
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="Paste a draft of your own writing to adapt."
            />
          </div>

          <div className="rounded-[var(--radius-md)] border border-[var(--border)]">
            <button
              type="button"
              aria-expanded={showKeyPanel}
              onClick={() => setShowKeyPanel((v) => !v)}
              className="flex w-full items-center justify-between px-3.5 py-2.5 text-left text-sm font-medium focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--accent)]"
            >
              <span>Optional: bring your own LLM key</span>
              <span
                className="font-mono text-xs text-[var(--muted-foreground)]"
                aria-hidden
              >
                {showKeyPanel ? "hide" : "show"}
              </span>
            </button>
            {showKeyPanel && (
              <div className="flex flex-col gap-3 border-t border-[var(--border)] px-3.5 py-4">
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="api-key">API key</Label>
                  <Input
                    id="api-key"
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="Stays in this session only"
                    autoComplete="off"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="base-url">
                    Base URL (OpenAI default, or OpenRouter)
                  </Label>
                  <Input
                    id="base-url"
                    value={baseUrl}
                    onChange={(e) => setBaseUrl(e.target.value)}
                    placeholder="https://api.openai.com/v1"
                  />
                </div>
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="model">Model name</Label>
                  <Input
                    id="model"
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    placeholder="gpt-4o-mini"
                  />
                </div>
              </div>
            )}
          </div>

          {error && <ErrorNote message={error} />}

          <div>
            <Button onClick={handleAdapt} disabled={loading || !profile}>
              {loading ? "Adapting" : "Adapt"}
            </Button>
          </div>
        </div>

        <div>
          {result ? (
            <Card className="vp-fade-up flex flex-col gap-6 p-6">
              <div className="grid grid-cols-2 gap-4">
                <Metric
                  label="Voice match before"
                  value={pct(result.voice_match_before)}
                />
                <Metric
                  label="Voice match after"
                  value={pct(result.voice_match_after)}
                  highlight={
                    result.voice_match_after >= result.voice_match_before
                  }
                />
              </div>
              {semantic !== null && (
                <p className="text-xs text-[var(--muted-foreground)]">
                  Meaning preserved vs your draft: {pct(semantic)}
                </p>
              )}
              {pathNote && (
                <p className="font-mono text-xs text-[var(--muted-foreground)]">
                  {pathNote}
                </p>
              )}

              <RadarChart
                series={[
                  {
                    label: "Your voice",
                    axes: result.feedback.radar.author,
                    color: "#6366F1",
                  },
                  {
                    label: "Before",
                    axes: result.feedback.radar.before,
                    color: "#9CA3AF",
                  },
                  {
                    label: "After",
                    axes: result.feedback.radar.after,
                    color: "#10B981",
                  },
                ]}
              />
            </Card>
          ) : (
            <Card className="flex h-full min-h-[20rem] items-center justify-center p-6">
              <p className="max-w-xs text-center text-sm text-[var(--muted-foreground)]">
                The voice match, adapted text, and per-sentence diff will appear
                here once you adapt a draft.
              </p>
            </Card>
          )}
        </div>
      </div>

      {result && (
        <div className="vp-fade-up mt-8 grid gap-8 lg:grid-cols-2">
          <div>
            <p className="mb-2 font-mono text-xs uppercase tracking-wide text-[var(--muted-foreground)]">
              Adapted draft
            </p>
            <Card className="whitespace-pre-wrap p-5 text-sm leading-relaxed">
              {result.adapted_text}
            </Card>
          </div>

          <div>
            <p className="mb-2 font-mono text-xs uppercase tracking-wide text-[var(--muted-foreground)]">
              Per-sentence diff
            </p>
            <Card className="p-5">
              {result.feedback.sentence_diff.length > 0 ? (
                <>
                  <div className="flex flex-wrap gap-2 leading-relaxed">
                    {result.feedback.sentence_diff.map((item, i) => (
                      <span
                        key={i}
                        title={`${DIFF_LABEL_TEXT[item.label]} - voice match ${pct(item.voice_match)}`}
                        className={`rounded-[var(--radius-sm)] border px-1.5 py-0.5 text-sm ${DIFF_STYLES[item.label]}`}
                      >
                        {item.sentence}
                      </span>
                    ))}
                  </div>
                  <DiffLegend />
                </>
              ) : (
                <p className="text-sm text-[var(--muted-foreground)]">
                  No per-sentence diff was returned for this draft.
                </p>
              )}
            </Card>
          </div>
        </div>
      )}
    </section>
  );
}

function Metric({
  label,
  value,
  highlight,
}: {
  label: string;
  value: string;
  highlight?: boolean;
}) {
  return (
    <div>
      <p className="font-mono text-[0.7rem] uppercase tracking-wide text-[var(--muted-foreground)]">
        {label}
      </p>
      <p
        className={`mt-1 text-2xl font-bold tracking-tight ${
          highlight ? "text-[var(--accent)]" : ""
        }`}
      >
        {value}
      </p>
    </div>
  );
}

function DiffLegend() {
  const items: Array<{ label: DiffLabel; dot: string }> = [
    { label: "improved", dot: "var(--accent)" },
    { label: "regressed", dot: "var(--destructive)" },
    { label: "same", dot: "var(--muted-foreground)" },
  ];
  return (
    <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 border-t border-[var(--border)] pt-4">
      {items.map((item) => (
        <span
          key={item.label}
          className="flex items-center gap-2 font-mono text-xs text-[var(--muted-foreground)]"
        >
          <span
            className="inline-block h-2.5 w-2.5 rounded-full"
            style={{ backgroundColor: item.dot }}
            aria-hidden
          />
          {DIFF_LABEL_TEXT[item.label]}
        </span>
      ))}
    </div>
  );
}
