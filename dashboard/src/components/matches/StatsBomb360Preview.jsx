"use client";

import { useEffect, useMemo, useState } from "react";
import { EmptyState } from "@/components/ui-custom/EmptyState";
import { ErrorState } from "@/components/ui-custom/ErrorState";
import { LoadingState } from "@/components/ui-custom/LoadingState";
import { useMatchFrames } from "@/hooks/useMatchFrames";

const PITCH_WIDTH = 120;
const PITCH_HEIGHT = 80;

export function StatsBomb360Preview({ matchId, has360 }) {
  const { frameData, loading, error } = useMatchFrames(
    matchId,
    Boolean(has360),
  );
  const frames = frameData?.frames || [];
  const [selectedIndex, setSelectedIndex] = useState(0);
  const selectedFrame = frames[selectedIndex] || null;

  useEffect(() => {
    setSelectedIndex(0);
  }, [matchId, frames.length]);

  if (!has360) {
    return (
      <section className="space-y-4">
        <div>
          <h2 className="text-xl font-semibold text-white">
            StatsBomb 360 pitch preview
          </h2>
          <p className="mt-1 text-sm text-muted">
            No 360 freeze frames are available for this match.
          </p>
        </div>
      </section>
    );
  }

  return (
    <section className="space-y-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">
            StatsBomb 360 pitch preview
          </h2>
          <p className="mt-1 text-sm text-muted">
            Preview freeze-frame player positions for this match.
          </p>
        </div>
        {selectedFrame ? (
          <div className="text-sm text-muted">
            Frame {selectedIndex + 1} of {frames.length}
          </div>
        ) : null}
      </div>

      {loading ? <LoadingState label="Loading 360 freeze frames" /> : null}
      {error ? (
        <ErrorState title="Could not load 360 frames" message={error} />
      ) : null}
      {!loading && !error && !frames.length ? (
        <EmptyState
          title="No freeze frames"
          message="StatsBomb did not return any 360 freeze frames for this match."
        />
      ) : null}
      {!loading && !error && selectedFrame ? (
        <div className="card p-5">
          <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_260px]">
            <PitchSvg frame={selectedFrame} />
            <FrameControls
              frame={selectedFrame}
              frames={frames}
              selectedIndex={selectedIndex}
              onSelect={setSelectedIndex}
            />
          </div>
        </div>
      ) : null}
    </section>
  );
}

function FrameControls({ frame, frames, selectedIndex, onSelect }) {
  const qualityClass =
    frame.visible_player_count >= 14 ? "text-emerald-300" : "text-amber-300";

  return (
    <div className="space-y-4">
      <div>
        <label
          htmlFor="frame-selector"
          className="text-sm font-medium text-slate-200"
        >
          Frame selector
        </label>
        <input
          id="frame-selector"
          type="range"
          min="0"
          max={Math.max(frames.length - 1, 0)}
          value={selectedIndex}
          onChange={(event) => onSelect(Number(event.target.value))}
          className="mt-3 w-full accent-sky-400"
        />
      </div>

      <div className="grid grid-cols-2 gap-3 text-sm">
        <Info label="Visible players" value={frame.visible_player_count} />
        <Info label="Frame ID" value={shortId(frame.id)} />
      </div>

      <div
        className={`rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm font-semibold ${qualityClass}`}
      >
        {frame.quality_label}
        {frame.visible_player_count < 14 ? (
          <div className="mt-1 text-xs font-normal text-muted">
            This frame has less than 14 visible players.
          </div>
        ) : null}
      </div>

      <div className="space-y-2 text-sm text-muted">
        <LegendDot
          color="#facc15"
          label="Actor"
          description={frame.actor_player_name || "Player doing the action in this frame"}
        />
        <LegendDot
          color="#38bdf8"
          label="Teammates"
          description={frame.actor_team_name || "Same team as the actor"}
        />
        <LegendDot
          color="#fb7185"
          label="Opponents"
          description={frame.opponent_team_name || "Opposing team"}
        />
        <LegendDot color="#c084fc" label="Goalkeeper" description="Special marker when keeper is true" />
      </div>
    </div>
  );
}

function PitchSvg({ frame }) {
  const visibleAreaPoints = useMemo(
    () => polygonPoints(frame.visible_area),
    [frame.visible_area],
  );
  const players = (frame.players || []).filter((player) =>
    validLocation(player.location),
  );

  return (
    <svg
      viewBox={`0 0 ${PITCH_WIDTH} ${PITCH_HEIGHT}`}
      className="w-full rounded-md border border-slate-700 bg-emerald-950"
    >
      <style>{`
        .frame-marker .marker-shape {
          transition: stroke-width 120ms ease, stroke 120ms ease, opacity 120ms ease;
        }
        .frame-marker:hover .marker-shape {
          stroke: #ffffff;
          stroke-width: 1.3;
          opacity: 1;
        }
        .frame-marker:hover .marker-hitbox {
          opacity: 0.12;
        }
      `}</style>
      <rect
        x="0"
        y="0"
        width={PITCH_WIDTH}
        height={PITCH_HEIGHT}
        fill="#064e3b"
      />
      <rect
        x="3"
        y="3"
        width="114"
        height="74"
        fill="none"
        stroke="#d1fae5"
        strokeWidth="0.8"
      />
      <line x1="60" y1="3" x2="60" y2="77" stroke="#d1fae5" strokeWidth="0.8" />
      <circle
        cx="60"
        cy="40"
        r="9.15"
        fill="none"
        stroke="#d1fae5"
        strokeWidth="0.8"
      />
      <circle cx="60" cy="40" r="0.9" fill="#d1fae5" />
      <rect
        x="3"
        y="18"
        width="18"
        height="44"
        fill="none"
        stroke="#d1fae5"
        strokeWidth="0.8"
      />
      <rect
        x="99"
        y="18"
        width="18"
        height="44"
        fill="none"
        stroke="#d1fae5"
        strokeWidth="0.8"
      />
      <rect
        x="3"
        y="30"
        width="6"
        height="20"
        fill="none"
        stroke="#d1fae5"
        strokeWidth="0.8"
      />
      <rect
        x="111"
        y="30"
        width="6"
        height="20"
        fill="none"
        stroke="#d1fae5"
        strokeWidth="0.8"
      />
      {visibleAreaPoints ? (
        <polygon
          points={visibleAreaPoints}
          fill="#f8fafc"
          opacity="0.16"
          stroke="#f8fafc"
          strokeWidth="0.7"
        />
      ) : null}
      {players.map((player, index) => (
        <PlayerMarker key={`${frame.id}-${index}`} player={player} />
      ))}
    </svg>
  );
}

function PlayerMarker({ player }) {
  const [x, y] = player.location;
  const title = markerTitle(player);

  if (player.actor) {
    return (
      <g className="frame-marker cursor-pointer">
        <title>{title}</title>
        <circle
          className="marker-hitbox"
          cx={x}
          cy={y}
          r="4.4"
          fill="#ffffff"
          opacity="0"
        />
        <circle
          className="marker-shape"
          cx={x}
          cy={y}
          r="2.2"
          fill="#facc15"
          stroke="#111827"
          strokeWidth="0.8"
        />
      </g>
    );
  }

  if (player.keeper) {
    return (
      <g className="frame-marker cursor-pointer">
        <title>{title}</title>
        <circle
          className="marker-hitbox"
          cx={x}
          cy={y}
          r="4.4"
          fill="#ffffff"
          opacity="0"
        />
        <rect
          className="marker-shape"
          x={x - 1.8}
          y={y - 1.8}
          width="3.6"
          height="3.6"
          fill="#c084fc"
          stroke="#111827"
          strokeWidth="0.7"
          transform={`rotate(45 ${x} ${y})`}
        />
      </g>
    );
  }

  return (
    <g className="frame-marker cursor-pointer">
      <title>{title}</title>
      <circle
        className="marker-hitbox"
        cx={x}
        cy={y}
        r="4.2"
        fill="#ffffff"
        opacity="0"
      />
      <circle
        className="marker-shape"
        cx={x}
        cy={y}
        r="1.8"
        fill={player.teammate ? "#38bdf8" : "#fb7185"}
        stroke="#111827"
        strokeWidth="0.6"
      />
    </g>
  );
}

function Info({ label, value }) {
  return (
    <div className="rounded-md border border-slate-700 bg-slate-950 p-3">
      <div className="text-xs uppercase tracking-wide text-muted">{label}</div>
      <div className="mt-1 text-sm font-semibold text-white">{value}</div>
    </div>
  );
}

function LegendDot({ color, label, description }) {
  return (
    <div className="flex items-start gap-2">
      <span
        className="mt-1 h-3 w-3 rounded-full"
        style={{ backgroundColor: color }}
      />
      <span>
        <span className="font-semibold text-slate-200">{label}</span>
        <span className="block text-xs text-slate-400">{description}</span>
      </span>
    </div>
  );
}

function polygonPoints(values) {
  if (!Array.isArray(values) || values.length < 6) return "";

  const points = [];
  for (let index = 0; index < values.length - 1; index += 2) {
    const x = Number(values[index]);
    const y = Number(values[index + 1]);
    if (Number.isFinite(x) && Number.isFinite(y)) {
      points.push(`${x},${y}`);
    }
  }

  return points.length >= 3 ? points.join(" ") : "";
}

function validLocation(location) {
  return (
    Array.isArray(location) &&
    location.length >= 2 &&
    Number.isFinite(Number(location[0])) &&
    Number.isFinite(Number(location[1]))
  );
}

function shortId(id) {
  return String(id || "N/A").slice(0, 8);
}

function markerTitle(player) {
  const role = player.actor
    ? "Actor"
    : player.keeper
      ? "Goalkeeper"
      : player.teammate
        ? "Teammate"
        : "Opponent";
  const lines = [
    `Role: ${role}`,
    `Player: ${player.player_name || "Not available in 360 frame data"}`,
    `Position: ${player.position || "Not available in 360 frame data"}`,
    `Team: ${player.team_name || "Not available"}`,
  ];

  if (player.event_type) {
    lines.push(`Action: ${player.event_type}`);
  }

  return lines.join("\n");
}
