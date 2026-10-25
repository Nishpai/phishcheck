import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Page on 8742, API on 8741.
export default defineConfig({
  plugins: [react()],
  server: { host: "127.0.0.1", port: 8742 },
});
