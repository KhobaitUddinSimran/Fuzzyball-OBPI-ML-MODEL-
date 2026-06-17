/**
 * JSDoc type reference for OBPI dashboard data.
 *
 * @typedef {{M1_SC:number,M2_OIRC:number,M3_BRPC:number,M4_OBR90:number,M5_RBTL:number,M6_RUP:number,M7_SCI:number,M8_LPC:number,M9_CBI:number}} MetricBreakdown
 * @typedef {{M1:number,M2:number,M3:number,M4:number,M5:number,M6:number,M7:number,M8:number,M9:number}} ShapBreakdown
 * @typedef {{spatial:number,movement:number,receiving:number,temporal:number}} DimensionScores
 * @typedef {{player_id:number,player_name:string,match_id:number,team_name:string,position:string,minutes?:number,obpi_score:number,percentile?:number,obpi_style?:string,score_band?:string,dimensions:DimensionScores,metrics:MetricBreakdown,shap?:ShapBreakdown,summary?:string}} AnalyzeResponse
 */

export const metricLabels = {
  M1_SC: "M1 Screening Coefficient",
  M2_OIRC: "M2 Off-Ball Impact Run Coefficient",
  M3_BRPC: "M3 Best Receiving Position Coefficient",
  M4_OBR90: "M4 Off-Ball Runs per 90",
  M5_RBTL: "M5 Receipts Between the Lines",
  M6_RUP: "M6 Receipts Under Pressure",
  M7_SCI: "M7 Space Creation Index",
  M8_LPC: "M8 La Pausa Coefficient",
  M9_CBI: "M9 Call-for-Ball Index"
};

export const metricGroups = {
  Spatial: ["M1_SC", "M7_SCI"],
  Movement: ["M2_OIRC", "M4_OBR90"],
  Receiving: ["M3_BRPC", "M5_RBTL", "M6_RUP"],
  Temporal: ["M8_LPC", "M9_CBI"]
};
