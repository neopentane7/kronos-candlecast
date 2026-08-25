import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { VitePWA } from "vite-plugin-pwa";

// Served from https://<user>.github.io/kronos-candlecast/, so every asset URL needs the
// repo name as a prefix. Getting this wrong produces a blank page that looks like a build
// failure but is really a 404 on the JS bundle.
const BASE = "/kronos-candlecast/";

export default defineConfig({
  base: BASE,
  plugins: [
    react(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["favicon.svg"],
      manifest: {
        name: "CandleCast — calibrated forecast cones",
        short_name: "CandleCast",
        description:
          "Calibrated probability cones for NSE equities. Research tool, not investment advice.",
        theme_color: "#0d1b1e",
        background_color: "#0d1b1e",
        display: "standalone",
        start_url: BASE,
        scope: BASE,
        icons: [
          { src: "icon-192.png", sizes: "192x192", type: "image/png" },
          { src: "icon-512.png", sizes: "512x512", type: "image/png" },
          { src: "icon-512.png", sizes: "512x512", type: "image/png", purpose: "maskable" },
        ],
      },
      workbox: {
        globPatterns: ["**/*.{js,css,html,svg,png,webmanifest}"],
        // Forecasts are static JSON that changes once a day. Serve the cached copy
        // immediately so an installed app opens offline, then refresh in the background.
        runtimeCaching: [
          {
            urlPattern: ({ url }) => url.pathname.includes("/data/"),
            handler: "StaleWhileRevalidate",
            options: {
              cacheName: "candlecast-forecasts",
              expiration: { maxEntries: 200, maxAgeSeconds: 60 * 60 * 24 * 14 },
            },
          },
        ],
      },
    }),
  ],
});
