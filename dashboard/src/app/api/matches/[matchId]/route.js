import { NextResponse } from "next/server";
import { isDevelopment, modelRequest } from "@/lib/api/modelClient";
import { getMockMatch } from "@/lib/mock/mockMatches";

export async function GET(_request, { params }) {
  const { matchId } = await params;

  try {
    const data = await modelRequest(`/matches/${matchId}`);
    return NextResponse.json(data);
  } catch (error) {
    const match = getMockMatch(matchId);

    if (isDevelopment() && match) {
      return NextResponse.json(match, { headers: { "x-mock-data": "true" } });
    }

    return NextResponse.json({ error: "Unable to load match details from the model service." }, { status: match ? 502 : 404 });
  }
}
