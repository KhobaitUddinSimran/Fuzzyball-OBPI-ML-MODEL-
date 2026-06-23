import { NextResponse } from "next/server";
import { isDevelopment, modelRequest } from "@/lib/api/modelClient";
import { getMockStatsBomb360MatchIds } from "@/lib/mock/mockMatches";

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const refresh = searchParams.get("refresh") === "true";

  try {
    const data = await modelRequest(`/statsbomb-360/match-ids?refresh=${refresh}`);
    return NextResponse.json(data);
  } catch (error) {
    if (isDevelopment()) {
      return NextResponse.json(getMockStatsBomb360MatchIds(), { headers: { "x-mock-data": "true" } });
    }

    return NextResponse.json({ error: "Unable to load StatsBomb 360 availability." }, { status: 502 });
  }
}
