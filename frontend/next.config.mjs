/** @type {import('next').NextConfig} */

// Sprint 6: el proxy /api/proxy/* vive en un route handler (app/api/proxy)
// que lee API_INTERNAL_URL en runtime: los rewrites de next.config se hornean
// en el build de standalone y no admiten env de runtime.
const nextConfig = {
  output: "standalone",
};

export default nextConfig;
