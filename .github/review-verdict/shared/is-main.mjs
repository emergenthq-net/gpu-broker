// "Is this module the process entrypoint?" — compared by REALPATH. TEC-284.
//
// The naive guard `import.meta.url === `file://${process.argv[1]}`` is false whenever the script is
// invoked through a path that is not its realpath (macOS /tmp -> /private/tmp, a symlinked checkout
// such as _mundonow/www). The CLI body then silently does not run and the process exits 0 — which
// verdict.mjs documents as CLEAN. Normalising both sides with realpath makes the guard path-independent.
import { realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export function isMain(metaUrl, argv1 = process.argv[1]) {
  if (!argv1) return false;
  try {
    return realpathSync(fileURLToPath(metaUrl)) === realpathSync(argv1);
  } catch {
    return false;
  }
}
