import nextra from "nextra";

const withNextra = nextra({
  theme: "nextra-theme-docs",
  themeConfig: "./theme.config.tsx",
});

const nextConfig = {
  reactStrictMode: true,
  output: "export",
  env: {
    GTM_ID: process.env.GTM_ID,
  },
  images: {
    formats: ["image/webp"],
    unoptimized: true,
  },
};

export default withNextra(nextConfig);