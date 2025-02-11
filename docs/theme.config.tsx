import React from "react";
// import LogoMark from "@/components/logo-mark";
import { DocsThemeConfig } from "nextra-theme-docs";

const config: DocsThemeConfig = {
  logo: (
    <span className="flex gap-x-8 items-center">
      <div className="flex">
        {/* <LogoMark /> */}
        <span className="ml-2 font-semibold">Menlo Ichigo</span>
      </div>
    </span>
  ),
  project: {
    link: "https://github.com/janhq/ichigo",
  },
  docsRepositoryBase: "https://github.com/janhq/ichigo/tree/main/docs",
  footer: {
    content: (
      <span>
        ©{new Date().getFullYear()} <span>Menlo Research Company.</span>
      </span>
    ),
  },
};

export default config;
