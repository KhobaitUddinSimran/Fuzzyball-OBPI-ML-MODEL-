import { NextResponse } from "next/server";
import { isDevelopment, modelRequest } from "@/lib/api/modelClient";
import { getMockMatchesByYear } from "@/lib/mock/mockMatches";

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const year = searchParams.get("year");

  if (!year) {
    return NextResponse.json({ error: "Missing required year query parameter." }, { status: 400 });
  }

  try {
    const data = await modelRequest(`/matches?event=fifa-world-cup&year=${encodeURIComponent(year)}`);
    return NextResponse.json(data);
  } catch (error) {
    if (isDevelopment()) {
      return NextResponse.json(getMockMatchesByYear(year), { headers: { "x-mock-data": "true" } });
    }

    return NextResponse.json({ error: "Unable to load matches from the model service." }, { status: 502 });
  }
}
