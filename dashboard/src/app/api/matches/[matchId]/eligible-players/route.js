import { NextResponse } from "next/server";
import { isDevelopment, modelRequest } from "@/lib/api/modelClient";
import { getMockEligiblePlayers } from "@/lib/mock/mockEligiblePlayers";

export async function GET(_request, { params }) {
  const { matchId } = await params;

  try {
    const data = await modelRequest(`/matches/${matchId}/eligible-players`);
    return NextResponse.json(data);
  } catch (error) {
    const players = getMockEligiblePlayers(matchId);

    if (isDevelopment() && players.length) {
      return NextResponse.json(players, { headers: { "x-mock-data": "true" } });
    }

    return NextResponse.json({ error: "Unable to load eligible players from the model service." }, { status: players.length ? 502 : 404 });
  }
}
