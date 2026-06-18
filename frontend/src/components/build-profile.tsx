"use client";

import { useState } from "react";
import {
  ApiError,
  AXIS_KEYS,
  AXIS_LABELS,
  createProfile,
  type StyleProfileResponse,
} from "@/lib/api";
import { RadarChart } from "@/components/radar-chart";
import { Button, Card, ErrorNote, Input, Label, Textarea } from "@/components/ui";

const SAMPLE_SLOTS = [
  { label: "Sample 1", optional: false },
  { label: "Sample 2", optional: false },
  { label: "Sample 3", optional: false },
  { label: "Sample 4", optional: true },
  { label: "Sample 5", optional: true },
];

// Replicates the Gradio normalize_axes scales so the built profile renders a
// fingerprint immediately, without a second round trip.
const RADAR_SCALE: Record<string, number> = {
  avg_sentence_length: 40,
  type_token_ratio: 1,
  contraction_rate: 0.1,
  passive_voice_ratio: 1,
  function_word_ratio: 0.6,
  transition_word_rate: 0.1,
};

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v));
}

function num(record: Record<string, unknown>, key: string): number {
  const v = record[key];
  return typeof v === "number" && Number.isFinite(v) ? v : 0;
}

function normalizeAxes(profile: StyleProfileResponse) {
  const ling = profile.linguistic_features ?? {};
  const sty = profile.stylometric_features ?? {};
  const axes: Record<string, number> = {};
  for (const key of AXIS_KEYS) {
    const source = key === "avg_sentence_length" || key === "type_token_ratio"
      ? ling
      : sty;
    axes[key] = clamp01(num(source, key) / (RADAR_SCALE[key] || 1));
  }
  return axes;
}

interface BuildProfileProps {
  onProfileBuilt: (profile: StyleProfileResponse) => void;
}

export function BuildProfile({ onProfileBuilt }: BuildProfileProps) {
  const [samples, setSamples] = useState<string[]>(["", "", "", "", ""]);
  const [profileName, setProfileName] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [profile, setProfile] = useState<StyleProfileResponse | null>(null);

  const filled = samples.map((s) => s.trim()).filter(Boolean);
  const totalWords = filled.reduce(
    (acc, s) => acc + s.split(/\s+/).filter(Boolean).length,
    0,
  );

  function setSample(index: number, value: string) {
    setSamples((prev) => {
      const next = [...prev];
      next[index] = value;
      return next;
    });
  }

  async function handleBuild() {
    setError(null);

    if (filled.length < 3) {
      setError("Please provide at least 3 writing samples.");
      return;
    }
    if (totalWords < 500) {
      setError(`Need at least 500 words total. You provided ${totalWords}.`);
      return;
    }

    setLoading(true);
    try {
      const result = await createProfile({
        user_id: "demo-user",
        samples: filled,
        profile_name: profileName.trim() || "My voice",
      });
      setProfile(result);
      onProfileBuilt(result);
    } catch (err) {
      setError(
        err instanceof ApiError
          ? err.message
          : "Something went wrong while building your profile.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <section id="build" className="scroll-mt-20">
      <div className="mb-8">
        <p className="font-mono text-xs tracking-wide text-[var(--accent)]">
          01 / Build your profile
        </p>
        <h2 className="mt-2 text-2xl font-bold tracking-tight">
          Measure your writing voice
        </h2>
        <p className="mt-2 max-w-2xl text-sm leading-relaxed text-[var(--muted-foreground)]">
          Paste at least three samples of your own writing, 500 or more words in
          total. You analyze your own writing here.
        </p>
      </div>

      <div className="grid gap-8 lg:grid-cols-2">
        <div className="flex flex-col gap-4">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="profile-name">Profile name (optional)</Label>
            <Input
              id="profile-name"
              value={profileName}
              onChange={(e) => setProfileName(e.target.value)}
              placeholder="My voice"
            />
          </div>

          {SAMPLE_SLOTS.map((slot, i) => (
            <div key={slot.label} className="flex flex-col gap-1.5">
              <Label htmlFor={`sample-${i}`}>
                {slot.label}
                {slot.optional ? " (optional)" : ""}
              </Label>
              <Textarea
                id={`sample-${i}`}
                rows={4}
                value={samples[i]}
                onChange={(e) => setSample(i, e.target.value)}
                placeholder="Paste a passage of your own writing."
              />
            </div>
          ))}

          <p className="font-mono text-xs text-[var(--muted-foreground)]">
            {filled.length} sample{filled.length === 1 ? "" : "s"} -{" "}
            {totalWords} word{totalWords === 1 ? "" : "s"}
          </p>

          {error && <ErrorNote message={error} />}

          <div>
            <Button onClick={handleBuild} disabled={loading}>
              {loading ? "Building" : "Build profile"}
            </Button>
          </div>
        </div>

        <div>
          {profile ? (
            <Card className="vp-fade-up flex flex-col gap-6 p-6">
              <RadarChart
                series={[
                  {
                    label: "Your voice",
                    axes: normalizeAxes(profile),
                    color: "#10B981",
                  },
                ]}
              />
              <ProfileSummary profile={profile} />
            </Card>
          ) : (
            <Card className="flex h-full min-h-[20rem] items-center justify-center p-6">
              <p className="max-w-xs text-center text-sm text-[var(--muted-foreground)]">
                Your voice fingerprint will appear here once you build a
                profile.
              </p>
            </Card>
          )}
        </div>
      </div>
    </section>
  );
}

function ProfileSummary({ profile }: { profile: StyleProfileResponse }) {
  const ling = profile.linguistic_features ?? {};
  const sty = profile.stylometric_features ?? {};
  const meta = (profile.embedding_metadata ?? {}) as Record<string, unknown>;
  const consistency =
    (meta.consistency_metrics as Record<string, unknown> | undefined) ?? {};
  const meanSim =
    typeof consistency.mean_pairwise_similarity === "number"
      ? consistency.mean_pairwise_similarity
      : null;

  const axes = AXIS_KEYS.map((key) => {
    const source =
      key === "avg_sentence_length" || key === "type_token_ratio" ? ling : sty;
    return { key, value: num(source, key) };
  });

  const stats: Array<{ label: string; value: string }> = [
    { label: "Samples", value: String(profile.num_samples) },
    { label: "Words", value: String(profile.total_words) },
    {
      label: "Avg sentence length",
      value: `${num(ling, "avg_sentence_length").toFixed(1)} words`,
    },
    {
      label: "Lexical diversity",
      value: num(ling, "type_token_ratio").toFixed(2),
    },
    {
      label: "Contraction rate",
      value: `${(num(sty, "contraction_rate") * 100).toFixed(1)}%`,
    },
  ];
  if (meanSim !== null) {
    stats.push({
      label: "Sample consistency",
      value: meanSim.toFixed(2),
    });
  }

  return (
    <div className="flex flex-col gap-5 border-t border-[var(--border)] pt-5">
      <dl className="grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-3">
        {stats.map((s) => (
          <div key={s.label}>
            <dt className="font-mono text-[0.7rem] uppercase tracking-wide text-[var(--muted-foreground)]">
              {s.label}
            </dt>
            <dd className="mt-0.5 text-sm font-medium">{s.value}</dd>
          </div>
        ))}
      </dl>

      <div>
        <p className="mb-2 font-mono text-[0.7rem] uppercase tracking-wide text-[var(--muted-foreground)]">
          Fingerprint axes (0 to 1)
        </p>
        <ul className="flex flex-col gap-2">
          {axes.map(({ key, value }) => {
            const normalized = clamp01(value / (RADAR_SCALE[key] || 1));
            return (
              <li key={key} className="flex items-center gap-3">
                <span className="w-32 shrink-0 text-xs text-[var(--muted-foreground)]">
                  {AXIS_LABELS[key]}
                </span>
                <span
                  className="h-1.5 flex-1 overflow-hidden rounded-full bg-[var(--muted)]"
                  aria-hidden
                >
                  <span
                    className="block h-full rounded-full bg-[var(--accent)]"
                    style={{ width: `${normalized * 100}%` }}
                  />
                </span>
                <span className="w-9 shrink-0 text-right font-mono text-xs">
                  {normalized.toFixed(2)}
                </span>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}
