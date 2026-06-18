"use client";

import { useEffect, useState } from "react";
import {
  API_URL,
  checkHealth,
  type StyleProfileResponse,
} from "@/lib/api";
import { BuildProfile } from "@/components/build-profile";
import { AdaptDraft } from "@/components/adapt-draft";

type BackendStatus = "checking" | "online" | "offline";

export default function Home() {
  const [profile, setProfile] = useState<StyleProfileResponse | null>(null);
  const [status, setStatus] = useState<BackendStatus>("checking");

  useEffect(() => {
    let active = true;
    checkHealth()
      .then(() => active && setStatus("online"))
      .catch(() => active && setStatus("offline"));
    return () => {
      active = false;
    };
  }, []);

  return (
    <div className="dot-grid-bg min-h-dvh">
      <main className="mx-auto max-w-5xl px-6 py-20 lg:px-8 lg:py-28">
        <header className="mb-16">
          <p className="font-mono text-xs tracking-wide text-[var(--accent)]">
            VoicePrint
          </p>
          <h1 className="mt-3 text-4xl font-bold tracking-[-0.02em] md:text-5xl">
            Measure your writing voice, then adapt your own drafts toward it.
          </h1>
          <p className="mt-4 max-w-2xl text-base leading-relaxed text-[var(--muted-foreground)]">
            VoicePrint reads a stylometric fingerprint from your own samples,
            scores how far a draft sits from it, and nudges the draft back. This
            is voice adaptation on your own writing.
          </p>

          <BackendNote status={status} />
        </header>

        <div className="flex flex-col gap-20">
          <BuildProfile onProfileBuilt={setProfile} />
          <hr className="border-[var(--border)]" />
          <AdaptDraft profile={profile} />
        </div>

        <footer className="mt-24 border-t border-[var(--border)] pt-8">
          <p className="text-sm text-[var(--muted-foreground)]">
            Built by{" "}
            <a
              href="https://github.com/Ab-Romia/VoicePrint"
              target="_blank"
              rel="noopener noreferrer"
              className="link-underline text-[var(--foreground)]"
            >
              Ab-Romia
            </a>
            .
          </p>
        </footer>
      </main>
    </div>
  );
}

function BackendNote({ status }: { status: BackendStatus }) {
  const dotColor =
    status === "online"
      ? "var(--accent)"
      : status === "offline"
        ? "var(--destructive)"
        : "var(--muted-foreground)";

  const text =
    status === "online"
      ? `Connected to the backend at ${API_URL}.`
      : status === "offline"
        ? "The backend is not reachable right now. This demo needs the VoicePrint backend running. See DEPLOY.md to run it locally."
        : "Checking the backend connection.";

  return (
    <div className="mt-8 inline-flex items-center gap-2.5 rounded-[var(--radius-md)] border border-[var(--border)] bg-[var(--card)] px-3.5 py-2.5">
      <span
        className="inline-block h-2 w-2 shrink-0 rounded-full"
        style={{ backgroundColor: dotColor }}
        aria-hidden
      />
      <span className="text-xs text-[var(--muted-foreground)]">{text}</span>
    </div>
  );
}
