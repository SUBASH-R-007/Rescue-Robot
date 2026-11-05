// Lets `node --test webapp/tests` (a folder argument) run every *.test.mjs in this folder:
// Node runs a folder argument as a CommonJS module, which resolves to this index.js.
// `node --test "webapp/tests/*.test.mjs"` (and pytest) run the files directly instead.
'use strict';
const { readdirSync } = require('node:fs');
const { join } = require('node:path');
const { pathToFileURL } = require('node:url');

for (const name of readdirSync(__dirname).filter((f) => f.endsWith('.test.mjs')).sort()) {
  import(pathToFileURL(join(__dirname, name)).href);
}
