import React from "react";
import ReactDOM from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import App from "./App";
import "./index.css";

// The data changes once a day and is served from a CDN, so refetching on every focus
// buys nothing and costs bandwidth on mobile.
const client = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 1000 * 60 * 30, refetchOnWindowFocus: false, retry: 1 },
  },
});

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <QueryClientProvider client={client}>
      <App />
    </QueryClientProvider>
  </React.StrictMode>,
);
