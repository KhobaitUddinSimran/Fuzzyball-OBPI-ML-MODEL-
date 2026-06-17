import { NextResponse } from "next/server";
import { modelRequest } from "@/lib/api/modelClient";

export async function GET() {
  try {
    return NextResponse.json(await modelRequest("/health"));
  } catch (error) {
    return NextResponse.json({ status: "unavailable", error: error.message }, { status: 503 });
  }
}
