/** @type {import('next').NextConfig} */

// Sprint 6: proxy en runtime para que el tier del usuario no quede horneado
// en el bundle (NEXT_PUBLIC_* se inlinea en build). El frontend reescribe
// /api/proxy/* a la API real en el servidor: el navegador nunca ve la URL
// del backend ni necesita CORS.
const API_INTERNAL = process.env.API_INTERNAL_URL || "http://localhost:8010";

const nextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/proxy/:path*",
        destination: `${API_INTERNAL}/:path*`,
      },
    ];
  },
};

export default nextConfig;
