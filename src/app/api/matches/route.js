import { NextResponse } from "next/server";
import { isDevelopment, modelRequest } from "@/lib/api/modelClient";
import { getMockMatchesByDate } from "@/lib/mock/mockMatches";

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const date = searchParams.get("date");

  if (!date) {
    return NextResponse.json({ error: "Missing required date query parameter." }, { status: 400 });
  }

  try {
    const data = await modelRequest(`/matches?event=fifa-world-cup&date=${encodeURIComponent(date)}`);
    return NextResponse.json(data);
  } catch (error) {
    if (isDevelopment()) {
      return NextResponse.json(getMockMatchesByDate(date), { headers: { "x-mock-data": "true" } });
    }

    return NextResponse.json({ error: "Unable to load matches from the model service." }, { status: 502 });
  }
}
