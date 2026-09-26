# Resolve UI

A replay console for one event. It is static and offline, with no build step.

## Run it

```bash
python -m resolve export-ui --config configs/montha.yaml   # writes ui/data/montha.js
python -m resolve ui                                         # serves ui/ on http://localhost:8765 and opens it
```

You can also open `ui/index.html` directly from disk. It loads its data through `<script>` tags, so no server is needed. The fonts ship in `ui/fonts/` (IBM Plex, SIL OFL). When Google Fonts is unreachable, the page falls back to those local files.

The file in `ui/data/montha.js` is a **design preview** until `export-ui` replaces it. Its map fields are placeholders, and a banner says so. Never record a demo while that banner is showing.

## Controls

| Key | Action |
| --- | --- |
| ← → | Previous or next run |
| Space | Play or pause the replay |
| E / R | Earned-scale tiles / raw 0.25° ensemble |
| O | Observed-rain outline on or off |
| Z | Event view or full analysis domain |

## Recording the demo video

Use 1920×1080, browser zoom 100%, light theme, and full screen (F11). Hide the cursor except when pointing.

A suggested order:
1. Start at the 20 Oct run in Event view.
2. Press Space and let it play to the 28 Oct run.
3. Stop on the 27 Oct run. Toggle R and then E to contrast raw pixels with earned tiles.
4. Switch the skill panel to Table.

## Files

- `index.html`, `styles.css`, `app.js`: the page.
- `geo/region.js`: base map built from Natural Earth 1:10m (public domain), with the current Indian state boundaries including Telangana.
- `data/*.js`: event files, following `SCHEMA.md`.
