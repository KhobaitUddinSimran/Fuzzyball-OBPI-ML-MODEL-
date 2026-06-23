import { NextResponse } from "next/server";
import { isDevelopment, modelRequest } from "@/lib/api/modelClient";
import { getMockAnalysis } from "@/lib/mock/mockAnalysis";

export async function POST(request) {
  const body = await request.json();

  if (!body?.match_id || !body?.player_id) {
    return NextResponse.json({ error: "match_id and player_id are required." }, { status: 400 });
  }

  const payload = {
    match_id: Number(body.match_id),
    player_id: Number(body.player_id),
    tier: body.tier || "open"
  };

  try {
    const data = await modelRequest("/analyze", {
      method: "POST",
      body: JSON.stringify(payload)
    });
    return NextResponse.json(data);
  } catch (error) {
    const analysis = getMockAnalysis(payload);

    if (isDevelopment() && analysis) {
      return NextResponse.json(analysis, { headers: { "x-mock-data": "true" } });
    }

    return NextResponse.json(
      { error: error.message || "Unable to analyze this player because the model service is unavailable." },
      { status: analysis ? 502 : 404 },
    );
  }
}
