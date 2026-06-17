import { NextResponse } from "next/server";
import { isDevelopment, modelRequest } from "@/lib/api/modelClient";
import { mockDates } from "@/lib/mock/mockDates";

export async function GET() {
  try {
    const data = await modelRequest("/events/fifa-world-cup/dates");
    return NextResponse.json(data);
  } catch (error) {
    if (isDevelopment()) {
      return NextResponse.json(mockDates, { headers: { "x-mock-data": "true" } });
    }

    return NextResponse.json({ error: "Unable to load FIFA World Cup dates from the model service." }, { status: 502 });
  }
}
