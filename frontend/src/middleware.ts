import { clerkMiddleware, createRouteMatcher } from "@clerk/nextjs/server";
import { NextResponse } from "next/server";

// Clerk se activa solo si hay claves; sin ellas el middleware es transparente.
const clerkEnabled = !!process.env.NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY;

const isPrivateRoute = createRouteMatcher(["/screener(.*)"]);

export default clerkEnabled
  ? clerkMiddleware(async (auth, req) => {
      if (isPrivateRoute(req)) {
        await auth.protect();
      }
    })
  : function transparent(_req: Request) {
      return NextResponse.next();
    };

export const config = {
  matcher: ["/((?!_next|.*\\..*).*)", "/"],
};
