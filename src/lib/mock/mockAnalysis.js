import { getMockEligiblePlayer } from "./mockEligiblePlayers";
import { getMockMatch } from "./mockMatches";
import { getScoreBand } from "@/lib/utils/scoreBands";

const styles = ["Elite Creator", "Tempo Controller", "Direct Runner", "Balanced", "Raw Runner"];

export function getMockAnalysis({ match_id, player_id }) {
  const player = getMockEligiblePlayer(match_id, player_id);
  const match = getMockMatch(match_id);

  if (!player || !match) {
    return null;
  }

  const seed = (Number(player_id) + Number(match_id)) % 100;
  const metrics = {
    M1_SC: value(seed, 21),
    M2_OIRC: value(seed, 31),
    M3_BRPC: value(seed, 43),
    M4_OBR90: value(seed, 53),
    M5_RBTL: value(seed, 61),
    M6_RUP: value(seed, 71),
    M7_SCI: value(seed, 83),
    M8_LPC: value(seed, 89),
    M9_CBI: value(seed, 97)
  };

  const dimensions = {
    spatial: average(metrics.M1_SC, metrics.M7_SCI),
    movement: average(metrics.M2_OIRC, metrics.M4_OBR90),
    receiving: average(metrics.M3_BRPC, metrics.M5_RBTL, metrics.M6_RUP),
    temporal: average(metrics.M8_LPC, metrics.M9_CBI)
  };

  const obpiScore = average(dimensions.spatial, dimensions.movement, dimensions.receiving, dimensions.temporal);

  return {
    player_id: player.player_id,
    player_name: player.player_name,
    match_id: Number(match_id),
    team_name: player.team_name,
    position: player.position,
    minutes: player.minutes,
    obpi_score: obpiScore,
    percentile: Math.min(99, Math.round(45 + obpiScore * 55)),
    obpi_style: styles[seed % styles.length],
    score_band: getScoreBand(obpiScore),
    dimensions,
    metrics,
    shap: {
      M1: round((metrics.M1_SC - 0.5) * 0.2),
      M2: round((metrics.M2_OIRC - 0.5) * 0.2),
      M3: round((metrics.M3_BRPC - 0.5) * 0.2),
      M4: round((metrics.M4_OBR90 - 0.5) * 0.2),
      M5: round((metrics.M5_RBTL - 0.5) * 0.2),
      M6: round((metrics.M6_RUP - 0.5) * 0.2),
      M7: round((metrics.M7_SCI - 0.5) * 0.2),
      M8: round((metrics.M8_LPC - 0.5) * 0.2),
      M9: round((metrics.M9_CBI - 0.5) * 0.2)
    },
    summary: `${player.player_name} shows a ${getScoreBand(obpiScore).toLowerCase()} OBPI profile in ${match.home_team} vs ${match.away_team}, with the strongest contribution coming from ${strongestDimension(dimensions)} actions.`
  };
}

function value(seed, multiplier) {
  return round(0.25 + (((seed * multiplier) % 70) / 100));
}

function average(...values) {
  return round(values.reduce((sum, item) => sum + item, 0) / values.length);
}

function round(valueToRound) {
  return Math.round(valueToRound * 100) / 100;
}

function strongestDimension(dimensions) {
  return Object.entries(dimensions).sort((a, b) => b[1] - a[1])[0][0];
}
