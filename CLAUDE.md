# Project Rules

## UI / CSS

- Always use CSS classes. Never use inline `style=` attributes in HTML-generating code.
- Before adding a new class, read the existing CSS file and reuse a class that already covers the same style. No duplicate classes.
- Define all colors, spacing, and typography in the `.css` file — not in Python, JS, or HTML strings.
- For dynamic state (e.g. good/warn/bad), use modifier classes (`.trend-good`, `.trend-bad`) — never compute a hex value and inject it inline.
- Plotly/canvas-rendered charts are the only exception: CSS cannot reach SVG. Use named Python constants there and add a comment explaining why.
- Establish the full CSS class system before writing HTML generators on any new UI project.
