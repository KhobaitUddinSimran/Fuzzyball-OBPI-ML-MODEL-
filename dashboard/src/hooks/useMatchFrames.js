"use client";

import { useEffect, useState } from "react";

const matchFramesCache = {
  framesByMatchId: {}
};

export function useMatchFrames(matchId, enabled = true) {
  const cachedFrames = matchId ? matchFramesCache.framesByMatchId[matchId] : null;
  const [data, setData] = useState(cachedFrames || null);
  const [loading, setLoading] = useState(Boolean(matchId && enabled) && !cachedFrames);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!matchId || !enabled) {
      setData(null);
      setLoading(false);
      setError("");
      return;
    }

    if (matchFramesCache.framesByMatchId[matchId]) {
      setData(matchFramesCache.framesByMatchId[matchId]);
      setLoading(false);
      setError("");
      return;
    }

    let active = true;
    setData(null);
    setLoading(true);
    setError("");

    fetchJson(`/api/matches/${matchId}/frames`)
      .then((frames) => {
        matchFramesCache.framesByMatchId[matchId] = frames;
        if (active) setData(frames);
      })
      .catch((requestError) => {
        if (active) setError(requestError.message);
      })
      .finally(() => {
        if (active) setLoading(false);
      });

    return () => {
      active = false;
    };
  }, [matchId, enabled]);

  return { frameData: data, loading, error };
}

async function fetchJson(url) {
  const response = await fetch(url);
  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(data?.error || "Unable to load StatsBomb 360 frames");
  }

  return data;
}
