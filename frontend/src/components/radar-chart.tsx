"use client";

import { AXIS_KEYS, AXIS_LABELS, type Axes } from "@/lib/api";

export interface RadarSeries {
  label: string;
  axes: Axes;
  color: string;
}

interface RadarChartProps {
  series: RadarSeries[];
  size?: number;
}

// Hand-rolled SVG radar so the bundle stays light. Draws the six normalized
// fingerprint axes (0 to 1) as overlaid polygons, mirroring the Gradio plot.
export function RadarChart({ series, size = 320 }: RadarChartProps) {
  const cx = size / 2;
  const cy = size / 2;
  const radius = size / 2 - 56;
  const axes = AXIS_KEYS;
  const n = axes.length;

  const angleFor = (i: number) => -Math.PI / 2 + (i * 2 * Math.PI) / n;

  const point = (value: number, i: number) => {
    const r = Math.max(0, Math.min(1, value)) * radius;
    const a = angleFor(i);
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)] as const;
  };

  const rings = [0.25, 0.5, 0.75, 1];

  return (
    <figure className="flex flex-col items-center gap-4">
      <svg
        width={size}
        height={size}
        viewBox={`0 0 ${size} ${size}`}
        role="img"
        aria-label={`Voice fingerprint radar: ${series
          .map((s) => s.label)
          .join(", ")}`}
        className="overflow-visible"
      >
        {/* Grid rings */}
        {rings.map((ring) => {
          const pts = axes
            .map((_, i) => point(ring, i).join(","))
            .join(" ");
          return (
            <polygon
              key={ring}
              points={pts}
              fill="none"
              stroke="var(--border)"
              strokeWidth={1}
            />
          );
        })}

        {/* Spokes and labels */}
        {axes.map((key, i) => {
          const [x, y] = point(1, i);
          const [lx, ly] = point(1.22, i);
          const anchor =
            Math.abs(lx - cx) < 2 ? "middle" : lx > cx ? "start" : "end";
          return (
            <g key={key}>
              <line
                x1={cx}
                y1={cy}
                x2={x}
                y2={y}
                stroke="var(--border)"
                strokeWidth={1}
              />
              <text
                x={lx}
                y={ly}
                textAnchor={anchor}
                dominantBaseline="middle"
                className="fill-[var(--muted-foreground)] font-mono"
                fontSize={11}
              >
                {AXIS_LABELS[key]}
              </text>
            </g>
          );
        })}

        {/* Series polygons */}
        {series.map((s) => {
          const pts = axes
            .map((key, i) => point(s.axes[key] ?? 0, i).join(","))
            .join(" ");
          return (
            <g key={s.label}>
              <polygon
                points={pts}
                fill={s.color}
                fillOpacity={0.12}
                stroke={s.color}
                strokeWidth={2}
              />
              {axes.map((key, i) => {
                const [px, py] = point(s.axes[key] ?? 0, i);
                return (
                  <circle
                    key={key}
                    cx={px}
                    cy={py}
                    r={2.5}
                    fill={s.color}
                  />
                );
              })}
            </g>
          );
        })}
      </svg>

      {series.length > 1 && (
        <figcaption className="flex flex-wrap items-center justify-center gap-x-5 gap-y-2">
          {series.map((s) => (
            <span
              key={s.label}
              className="flex items-center gap-2 font-mono text-xs text-[var(--muted-foreground)]"
            >
              <span
                className="inline-block h-2.5 w-2.5 rounded-full"
                style={{ backgroundColor: s.color }}
                aria-hidden
              />
              {s.label}
            </span>
          ))}
        </figcaption>
      )}
    </figure>
  );
}
