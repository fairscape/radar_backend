# Changelog: radar_backend

Every update of the live site (https://bioxplorer.org; https://radar.bioxplorer.org before 2026-11), newest first.
The frontend has its own changelog (`radar_frontend/CHANGELOG.md`); the two
are deployed together, and each entry names the frontend commit that went
live with it.

**Branches.** `main` is what runs online; `develop` holds changes that have
been reviewed, tested and tried on the test site but are not online yet.
An update online fast-forwards `main` to `develop`, tags it `vYYYY.MM.DD`,
and adds an entry here.

Each entry has:
- **For users**: what changed on the site.
- **Backend**: what changed in this repository.
- **Operations**: migrations, settings, data files, and anything to do on
  the server.

Times are UTC. Entries before 2026-11-16 were written afterwards from the
deploy logs. The code went online from the working copy and was committed
soon after, so for those entries "commit" means the first commit that
contains the change.

---

## v2026.11.17 (2026-11-17 19:18): the site is https://bioxplorer.org
Frontend: `v2026.11.17` (docs only).

**For users**
- The site moved from https://radar.bioxplorer.org to **https://bioxplorer.org**. The old Vercel page that used to be there is gone.
- radar.bioxplorer.org and www.bioxplorer.org were sent on to the new name at first (19:18), then retired: the tunnel no longer routes them, and their DNS records are removed.
- Plain http goes to https.
- Everyone signs in once more, because cookies belong to one name. Data and accounts are unchanged, since accounts go by email.
- "Sign in with Prosopia" works again once Prosopia lists the new callback.

**Backend**
- No application code changed.
- `deploy/Caddyfile`: plain http (`X-Forwarded-Proto: http`) gets a 308 to https://bioxplorer.org.
- `deploy/cloudflared-config.yml` routes only `bioxplorer.org`.
- `radar.sh status`:
  - checks https://bioxplorer.org;
  - has a new "write origin" line: the backend takes writes only from its callback's origin, so it shows whether `.env` names the right site.
- `deploy/env.example` and its test: the callback is `https://bioxplorer.org/auth/login/callback`.
- `deploy/README.md` has a new section, "The site's name".

**Operations**
- No migration.
- `.env`: `BIOXPLORER_OIDC_REDIRECT_URI=https://bioxplorer.org/auth/login/callback`. That is the only change.
- DNS: `bioxplorer.org` is a CNAME to the tunnel (`cloudflared tunnel route dns -f radar bioxplorer.org`).
- Google has the new callback registered. Prosopia does not yet; Nathan is adding it (record 38).
- Record 38 has the steps and scripts (`staging-20261117/`).

## v2026.11.16 (2026-11-16 21:11)
Frontend: `v2026.11.16`.

**For users**
- *Edit interest → Topics*, on an interest saved before 2026-11-12:
  - BioXplorer first reads the seed papers again ("Extracting UMLS concepts: x of y seeds read…");
  - then it shows the whole topic list at once;
  - switching a topic on or off is immediate. It used to take about 30 s per click while the list kept growing.
- A UMLS topic that no seed brings any more now says why:
  - "its seed paper was removed", when you removed that seed in this edit;
  - "no seed brings it now", when the seed was read again or OpenAlex classed it again.
- A new public page, `/about`. The privacy page now also mentions the web server's access log.

**Backend**
- Building a topic list or saving a topic switch no longer reads seeds inside the request. Seeds are read beforehand through `POST …/umls`; a seed not read yet counts as pending.
- `unmatched_why` on topics (`seed_removed` | `no_seed`).
- The UMLS topic merge now dedupes by topic code (URL or bare id), with the name as a second key; a seed is no longer counted twice. Topic names come from the current name index.
- Name index:
  - `scripts/build_topic_index.py --topics-json <rich json>` rebuilds the name index from the rich index's topics, names only, with no OpenAlex calls.
  - At startup, the backend logs `api.umls_topic_names_differ` if the two indexes disagree, and `/api/health` reports `umls_topic_index: names_differ`.

**Operations**
- No migration.
- The name index in `data/umls_cache` was rebuilt from the rich index, so 56 renamed topics now match. Restart the backend after any rebuild.

## 2026-11-12 07:34: UMLS topics from the sentence a concept is read in
Commits `0d36c99` (docs `29896eb`). Frontend `33bcede`.

**For users**
- UMLS topics are chosen from the sentence where the paper uses the concept, and they show that word: from "word" (UMLS: concept).
- Each concept gives at most one topic: the nearest of 3 neighbours that the interest does not already have. There are at most 5 per seed and 20 per interest, and UMLS topics start off.
- Removing a seed never drops a saved topic. Topics no seed brings any more stay as they were, marked.

**Operations**
- No migration.
- Needs the rich topic index (`openalex_topic_rich_*`) in `data/umls_cache`; see `docs/UMLS_TOPICS.md`.

## 2026-11-11 22:01: Ollama on CUDA (operations only)
- The live Ollama moved from Vulkan to CUDA: an embedding call went from 2.08 s to 0.13 s.
- No code change.

## 2026-11-11 02:43: frontend review fixes
Commit `127cddd`. Frontend `476831c`.
- `/api/users/me/prosopia` recognises a Prosopia profile already imported through the same ORCID.
- The rest of this update is frontend; see its changelog.
- No migration.

## 2026-11-07 05:54: UMLS similarity bar
Commit `4c8a052`.
- A UMLS concept gives a topic only when the concept and the topic are at least 0.80 alike. When one falls below, the next one takes its place.
- Score-bar settings must be between 0 and 1.
- No migration.

## 2026-11-06 16:49: switched-off topics hide their papers
Commit `4c8a052`. Frontend `8f3c7c5`.
- A paper leaves the feed when every topic that brought it is switched off. It comes back when one is switched on again; nothing is deleted.
- Research topics you cross out can be deleted for good.
- Text that looks like an API key is never stored.
- No migration.

## 2026-11-05 20:13: fixes from the 2026-11-05 review
Commit `4a09251`. Frontend `73ff9f1`.
- Fixes to research topics in drafts, the one-tab rule, expired edit copies, temporary files, "at least one topic", and key scrubbing in 401/403 errors.
- No migration.

## 2026-11-05 05:27: research topics on new-interest step 3
- Frontend only (`73ff9f1`).

## 2026-11-05 03:01: OpenAlex API key
Commit `4a09251`.
- With `RADAR_OPENALEX_API_KEY` set, every OpenAlex request carries the key, and error texts never show it.
- No migration.

## 2026-11-05 01:02: no topics guessed from unpublished PDFs
Commit `4a09251`. Frontend `73ff9f1`.
- Uploads no longer guess OpenAlex topics from a PDF's content.
- Topics guessed earlier are ignored everywhere and marked "PDF (old)" in saved interests.
- No migration.

## 2026-11-04 21:21: one tab changes an account
Commit `4a09251`. Frontend `73ff9f1`.
- The first tab holds the right to change things; other tabs are view-only, and their writes get 409.
- Closing that tab, or signing out in it, hands the right on.
- **Migration 0042** (`write_leases`).

## 2026-11-04 18:47: frontend review batch 1
- Frontend only (`73ff9f1`).

## 2026-11-04 06:32: abstracts headed "Abstract |"
Commit `91f1573`.
- PDF reading finds an abstract headed "Abstract |". When no end mark follows, the abstract ends at the paragraph break.
- No migration.

## 2026-11-04 05:02: a new interest's draft opens in one tab at a time
Commit `e10f505`. Frontend `b5301d5`.
- Another tab, Resume included, is refused.
- No migration.

## 2026-11-04 02:18: upload times on seeds, deleted marks
Commit `77a27d0`. Frontend `febd4eb`.
- Seeds show when their PDF was uploaded, and seeds whose paper was deleted are marked.
- **Migration 0041**.

## 2026-11-03 21:24: Prosopia's paged lists; Prosopia sign-in on
Commit `77a27d0`, from nsheff's 5905b16.
- Reads Prosopia's new `{results, pagination}` lists, and looks up an ORCID with one request.
- Prosopia sign-in turned on.
- No migration.

## 2026-11-03 18:58: sign in with Google or Prosopia
Commit `77a27d0`, from nsheff's backend PR #2. Frontend `febd4eb`.
- **Sign-in**:
  - Sign-in with Google or Prosopia, run by the backend over OpenID Connect, replaces Cloudflare Access. Anyone can sign in.
  - Identity is a signed session cookie only: Caddy strips `X-User-Email` and the `Cf-Access-*` headers, and the backend refuses cross-site writes.
  - CORS answers only the listed origins.
  - The backend refuses to start with unsafe settings.
- **Post-launch review fixes** CR.1–CR.11 for uploads and the Vault, scans, imports, deployment and the API.
- **Operations**:
  - **Migration 0040**.
  - New settings `BIOXPLORER_*` and `RADAR_TRUST_EMAIL_HEADER`; the session key must be at least 32 characters. See `deploy/env.example`.
  - New dependency: PyJWT.
  - The Caddyfile changed.

## 2026-11-02 15:06: online with a new, empty database
Commit `0cca0c5`. Frontend `54a6976`.
- The site switched to `functional-spec-oct2026` with an empty database. The old data was archived, not migrated.
- New since the previous site:
  - a draft sandbox for new interests;
  - scans stop when OpenAlex is unavailable;
  - private upload versions;
  - Edit interest works on a copy;
  - UMLS topics and the seed-removal rules;
  - adding seeds from an interest's feed;
  - per-user researcher imports, with an optional researcher profile;
  - duplicate seeds found by title and current OpenAlex ids;
  - a per-interest Saved list;
  - papers stay in the Vault until deleted from the Vault;
  - deleting an uploaded PDF;
  - a cross-user check on every API route.
