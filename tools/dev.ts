/** Dev server: watch, rebuild the web output, serve dist/ with live reload.
 *
 *   npm run dev            # http://localhost:8787
 */

import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

import { runDev } from "penname";

runDev({
  root: join(dirname(fileURLToPath(import.meta.url)), ".."),
  watch: ["paper", "src", "styles", "tools/build.ts"],
  buildCommand: ["node_modules/.bin/tsx", "tools/build.ts", "--no-pdf"],
});
