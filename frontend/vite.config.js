import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The page is http://127.0.0.1:8742. The API is a separate process on 8741.
export default defineConfig({
  plugins: [react()],
  server: {
    host: "127.0.0.1",
    port: 8742,
    strictPort: true,
  },
});
