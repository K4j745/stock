# `docs/data/` — generated dataset & large-file strategy

Everything under `docs/data/` is **generated output**, not hand-authored source.
It is produced by `python dashboard/generate.py` and refreshed on a schedule by
the *Update Dashboard Data* GitHub Actions workflow (`.github/workflows/dashboard.yml`,
Mon–Fri 17:00 UTC). GitHub Pages serves this folder statically; the frontend
`fetch()`es the JSON at runtime.

## Why not Git LFS?

Git LFS was considered for the large files here and **deliberately rejected** — it
would break the live site:

1. **GitHub Pages serves LFS *pointer files*, not the real content.** A page doing
   `fetch('data/signals/history.json')` would receive a ~130-byte text pointer
   instead of JSON, so every data-driven view would fail to parse.
2. **LFS bandwidth is metered** (1 GB/month on the free tier). A multi-MB file
   fetched by each visitor would exhaust the quota quickly and then return 429s.
3. LFS solves *repo cloning* weight, not *runtime delivery* weight — the actual
   problem here is delivery, and Pages + LFS are fundamentally incompatible.

So the fix is to **make the data small and to stop committing what nothing reads**,
not to move it into LFS.

## What was done (size + conflict reduction)

- **Dropped the unpublished per-ticker / per-model splits.**
  `signals/by_ticker/*.json` and `signals/by_model/*.json` (~165 MB combined) are
  **not fetched or linked by any page** — the frontend reads only `signals/latest.json`,
  `signals/history.json`, and `exports/all_signals.csv`. `write_signals_bundle()`
  now skips them by default (`write_splits=False`); pass `write_splits=True` to
  reconstruct them locally. They are also git-ignored so a local run can't re-add them.
- **Marked the whole tree `linguist-generated`** (`.gitattributes`) so reviews
  collapse the noisy byte diffs and language stats ignore them.
- **Added a `merge=keep-ours` driver** for `docs/data/**` so syncing the default
  branch into a feature branch never stops on a meaningless data conflict
  (whichever side ran `generate.py` last is authoritative). Enable once per clone:

  ```bash
  git config merge.keep-ours.name  "Keep our generated data on merge"
  git config merge.keep-ours.driver "true"
  ```

## Why merge conflicts kept reappearing (and how to avoid them)

The default branch gets a **daily bot commit** that rewrites `docs/data/**`. Any PR
that also regenerates that data will conflict with each new bot commit until it
merges. Practical guidance:

- **Merge data PRs promptly** — before the next scheduled bot run (Mon–Fri 17:00 UTC).
- With the `merge=keep-ours` driver configured, `git merge origin/<default-branch>`
  into your feature branch resolves the data automatically; only real code changes
  can then conflict.

## Recommended long-term upgrade: generate at deploy time

The cleanest end state is to **stop committing generated data entirely** and build it
during deployment, which removes the conflict class *and* the repo bloat for good:

1. In repo **Settings → Pages → Build and deployment**, set **Source = GitHub Actions**.
2. Change the workflow to run `python dashboard/generate.py`, then publish `docs/`
   with `actions/upload-pages-artifact` + `actions/deploy-pages` — **without**
   `git add docs/ && git commit`.
3. Add `docs/data/` to `.gitignore`.

Result: the site still updates on schedule, but no dataset ever lands in git, so
there are zero data merge conflicts and no large blobs in history. (`history.json`
can additionally be windowed to a bounded recent range in `write_signals_bundle()`
for faster client loads, with the full series still available via `all_signals.csv`.)
