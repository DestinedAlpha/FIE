import { defineConfig, devices } from "@playwright/test";
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false,
  workers: 1,
  timeout: 30000,
  use: {
    baseURL: "http://127.0.0.1:8000",
    ...devices["Desktop Chrome"],
    screenshot: "only-on-failure",
  },
  reporter: [["list"]],
  projects: [
    { name: "desktop" },
    {
      name: "mobile",
      use: { ...devices["iPhone 13"], defaultBrowserType: "chromium" },
    },
  ],
});
