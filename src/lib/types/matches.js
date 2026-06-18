/**
 * JSDoc type reference for the FIFA World Cup match workflow.
 *
 * @typedef {{year:string,season_id:number,label:string}} MatchYear
 * @typedef {{match_id:number,date:string,competition:string,stage?:string,home_team:string,away_team:string,home_score?:number,away_score?:number,kickoff_time?:string,stadium?:string}} MatchSummary
 * @typedef {{team_id:number,team_name:string}} TeamInfo
 * @typedef {MatchSummary & {eligible_player_count?:number,teams:{home:TeamInfo,away:TeamInfo}}} MatchDetails
 * @typedef {{player_id:number,player_name:string,team_id:number,team_name:string,position:string,minutes?:number}} EligiblePlayer
 */

export const fifaWorldCupEvent = {
  id: "fifa-world-cup",
  label: "FIFA World Cup"
};
