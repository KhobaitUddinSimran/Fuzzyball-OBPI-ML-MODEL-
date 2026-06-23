"use client";

import { useEffect, useState } from "react";

const matchPlayersCache = {
  playersByMatchId: {}
};

export function useEligiblePlayers(matchId) {
  const cachedPlayers = matchId ? matchPlayersCache.playersByMatchId[matchId] : null;
  const [data, setData] = useState(cachedPlayers || []);
  const [loading, setLoading] = useState(Boolean(matchId) && !cachedPlayers);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!matchId) {
      setData([]);
      setLoading(false);
      setError("");
      return;
    }

    if (matchPlayersCache.playersByMatchId[matchId]) {
      setData(matchPlayersCache.playersByMatchId[matchId]);
      setLoading(false);
      setError("");
      return;
    }

    let active = true;
    setData([]);
    setLoading(true);
    setError("");

    fetchJson(`/api/matches/${matchId}/eligible-players`)
      .then((players) => {
        matchPlayersCache.playersByMatchId[matchId] = players;
        if (active) setData(players);
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
  }, [matchId]);

  return { players: data, loading, error };
}

async function fetchJson(url) {
  const response = await fetch(url);
  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(data?.error || "Unable to load StatsBomb player data");
  }

  return data;
}
