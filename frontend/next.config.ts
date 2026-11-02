import type { NextConfig } from "next";

const backend = process.env.NEXUS_API_URL ?? "http://127.0.0.1:8100";

const config: NextConfig = {
  output: "standalone",
  compress: false, // gzip would buffer the chat event stream
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${backend}/api/:path*` }];
  },
};

export default config;
