"use client";

import { useEffect, useState } from "react";

const worldCupCache = {
  years: null,
  matchesByYear: {},
  matchDetailsById: {},
  statsBomb360: null
};

export function useWorldCupYears() {
  const [data, setData] = useState(worldCupCache.years || []);
  const [loading, setLoading] = useState(!worldCupCache.years);
  const [error, setError] = useState("");

  useEffect(() => {
    if (worldCupCache.years) {
      setData(worldCupCache.years);
      setLoading(false);
      setError("");
      return;
    }

    let active = true;
    setLoading(true);
    setError("");

    fetchJson("/api/events/fifa-world-cup/years")
      .then((years) => {
        worldCupCache.years = years;
        if (active) setData(years);
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
  }, []);

  return { years: data, loading, error };
}

export function useWorldCupMatches(year) {
  const cachedMatches = year ? worldCupCache.matchesByYear[year] : null;
  const [data, setData] = useState(cachedMatches || []);
  const [loading, setLoading] = useState(Boolean(year) && !cachedMatches);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!year) {
      setData([]);
      setLoading(false);
      setError("");
      return;
    }

    if (worldCupCache.matchesByYear[year]) {
      setData(worldCupCache.matchesByYear[year]);
      setLoading(false);
      setError("");
      return;
    }

    let active = true;
    setData([]);
    setLoading(true);
    setError("");

    fetchJson(`/api/matches?year=${encodeURIComponent(year)}`)
      .then((matches) => {
        worldCupCache.matchesByYear[year] = matches;
        if (active) setData(matches);
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
  }, [year]);

  return { matches: data, loading, error };
}

export function useWorldCupMatch(matchId) {
  const cachedMatch = matchId ? worldCupCache.matchDetailsById[matchId] : null;
  const [data, setData] = useState(cachedMatch || null);
  const [loading, setLoading] = useState(Boolean(matchId) && !cachedMatch);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!matchId) {
      setData(null);
      setLoading(false);
      setError("");
      return;
    }

    if (worldCupCache.matchDetailsById[matchId]) {
      setData(worldCupCache.matchDetailsById[matchId]);
      setLoading(false);
      setError("");
      return;
    }

    let active = true;
    setData(null);
    setLoading(true);
    setError("");

    fetchJson(`/api/matches/${matchId}`)
      .then((match) => {
        worldCupCache.matchDetailsById[matchId] = match;
        if (active) setData(match);
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

  return { match: data, loading, error };
}

export function useStatsBomb360Availability() {
  const [data, setData] = useState(worldCupCache.statsBomb360);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function refresh() {
    setLoading(true);
    setError("");

    try {
      const availability = await fetchJson("/api/statsbomb-360/match-ids?refresh=true");
      worldCupCache.statsBomb360 = availability;
      apply360AvailabilityToCachedMatches(availability.match_ids || []);
      setData(availability);
      return availability;
    } catch (requestError) {
      setError(requestError.message);
      return null;
    } finally {
      setLoading(false);
    }
  }

  return { availability: data, refresh, loading, error };
}

function apply360AvailabilityToCachedMatches(matchIds) {
  const ids = new Set(matchIds.map((matchId) => Number(matchId)));

  Object.keys(worldCupCache.matchesByYear).forEach((year) => {
    worldCupCache.matchesByYear[year] = worldCupCache.matchesByYear[year].map((match) => ({
      ...match,
      has_360: ids.has(Number(match.match_id))
    }));
  });
}

async function fetchJson(url) {
  const response = await fetch(url);
  const data = await response.json().catch(() => null);

  if (!response.ok) {
    throw new Error(data?.error || "Unable to load StatsBomb data");
  }

  return data;
}
