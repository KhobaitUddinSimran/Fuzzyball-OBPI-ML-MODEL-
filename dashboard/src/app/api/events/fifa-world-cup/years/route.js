import { NextResponse } from "next/server";
import { isDevelopment, modelRequest } from "@/lib/api/modelClient";
import { mockYears } from "@/lib/mock/mockYears";

export async function GET() {
  try {
    const data = await modelRequest("/events/fifa-world-cup/years");
    return NextResponse.json(data);
  } catch (error) {
    if (isDevelopment()) {
      return NextResponse.json(mockYears, { headers: { "x-mock-data": "true" } });
    }

    return NextResponse.json({ error: "Unable to load FIFA World Cup years from the model service." }, { status: 502 });
  }
}
