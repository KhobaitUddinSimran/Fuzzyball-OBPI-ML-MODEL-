import { NextResponse } from "next/server";
import { isDevelopment, modelRequest } from "@/lib/api/modelClient";
import { getMock360Frames } from "@/lib/mock/mock360Frames";

export async function GET(_request, { params }) {
  const { matchId } = await params;

  try {
    const data = await modelRequest(`/matches/${matchId}/frames`);
    return NextResponse.json(data);
  } catch (error) {
    const frames = getMock360Frames(matchId);

    if (isDevelopment() && frames.frames.length) {
      return NextResponse.json(frames, { headers: { "x-mock-data": "true" } });
    }

    return NextResponse.json({ error: "Unable to load StatsBomb 360 frames from the model service." }, { status: 502 });
  }
}
