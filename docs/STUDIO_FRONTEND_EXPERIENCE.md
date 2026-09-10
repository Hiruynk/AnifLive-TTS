# Studio frontend experience

The Studio remains a local voice workstation with a dark wine/lilac palette,
editorial headings, its existing Overview film, and selective elevated surfaces.
All development and verification for this change runs in Docker. The backend
API, computation defaults, quality gates and scheduling behavior are unchanged.

## Design references

- [BoardUI foundations](https://www.boardui.com/getting-started) and
  [Data Table](https://www.boardui.com/components/data-table): consistent control
  geometry, clear primary/secondary information, quiet row separators and an
  unmistakable selected row.
- [BoardUI Dropdown](https://www.boardui.com/components/dropdown): grouped
  actions, clear focus/selection and short popover transitions.
- [ThreeUI](https://threeui.com/): restrained material depth and interaction
  feedback on existing controls. No persistent WebGL scene or copied template
  was introduced.

The original dock stays in place without an additional hamburger control.
Compound file/folder fields have one surface with a transparent editable
interior. The synthesis editor remains unframed. Motion describes opening,
closing, pressing and selection; it does not delay API execution.

## Feature and presentation map

The complete pre-change control attributes and API target inventory are retained
in `tests/fixtures/studio-ui-contract.json`. Regression tests require every
original control except the explicitly removed redundant navigation toggle, with
unchanged IDs, values, limits and data-action attributes.

| Area | Presentation | Retained behavior |
| --- | --- | --- |
| Overview | Existing film, product title and operational summary | Existing creation and navigation entry points; true runtime/job state |
| Synthesis | Playback before expression details at all widths; expandable generation/A-B controls | Text annotation, expressions, language/model, segments, playback, cancellation, download, metrics and history |
| Expressions | Voice direction and metadata grouped behind a labelled disclosure | Reference analysis, editing, qualification, promotion and deletion |
| Datasets / TSE | Compact scrollable stage strip; project collection first on mobile with inspector/list navigation | Import, review, annotations, playback, separation, splits and verification |
| Training | Existing real lifecycle, selected experiment and available actions | Training, reference decisions, holdout and production controls and evidence |
| Evaluation | Selected workflow and complete report/list views; grouped benchmark inputs | Baseline selection, listening, quality/performance reports and qualifications |
| Models / Engines / Jobs | Consistent selected rows, badges, readable errors and local table scrolling | All existing inspection, activation, promotion and job controls |
| Settings | Common defaults visible, advanced settings expandable, explicit unsaved state | Same seven-field PATCH contract and validation; edits survive polling and edits made during save |
| Shared dialogs | Existing controls with visible focus, validation and transition feedback | Escape/close behavior, path picker, all project/job fields |
| Languages / PWA | New labels in English, Traditional and Simplified Chinese; asset version updated | Existing locale choices and offline shell; no private runtime data cached |

Only the interface locale is persisted. Disclosure choices, page positions and
draft settings remain in the current document memory, not browser storage.
Disclosure state, input values and page reading positions survive presentation
changes within the page. Reloading starts the optional detail sections closed. Important errors remain visible until dismissed. The mobile layout
uses the existing bottom navigation and More sheet.

## Interaction and motion contract

Controls have a 44 px minimum primary touch target. Mobile project collections
precede their complete inspector; both remain reachable without deleting any
fields. Synthesis keeps the existing parent/iframe height and message protocol.

Disclosures use interruptible height/opacity transitions; popovers transition
both in and out. Competing fill-mode animations are removed from those
popovers. Pressed/selected states use short color, border and position feedback.
Reduced-motion mode disables these transitions. Embedded synthesis changes are
scoped to `body.embedded`; the standalone WebUI is retained.

## Verification

- Existing frontend/API regression tests plus the control/API preservation tests.
- Docker browser coverage for all 11 modules at 1440×900, 1280×720, 390×844,
  360×800, 844×390 and 768×1024.
- Interaction checks for hidden-field validation, settings polling/save races,
  duplicate submit, navigation position, Escape, mobile More and synthetic
  audio playback/download.
- Measured intermediate animation frames, rapid reversal, reduced motion,
  and computed transparent backgrounds for every path-picker input.
- Empty and large-list states, persistent error rendering, localization, and
  standalone synthesis compatibility.

Reusable browser checks are in `tests/browser`. Their mutation-capable tests
require the synthetic-fixture response header and refuse a real Studio endpoint.
The fixture uses temporary isolated metadata and synthetic PCM. It is not a
neural quality or latency benchmark and does not operate on production Run 10.

Computer use on the actual Studio at port 9891 verified the 2026-09-08
record-disclosure changes at desktop 1440×900 and 1280×720, phone-size 390×844
and 360×800, and landscape 844×390. All seven record sections were opened and
closed; the real 223-output training list scrolled to its final record inside a
bounded panel. Space/Enter operation and in-page navigation state were checked.
The Overview Job flow panel has four matching 10 px corners. These are browser
viewport simulations, not physical-phone tests, and do not certify neural
execution or unrelated end-to-end release gates.

## Long record sections and release boundaries

Worker events and verified outputs in Training and Evaluation, TSE events, Jobs
events, and model artifact lineage start collapsed. Click the labelled header
to open or close; the count reflects currently rendered entries. The bounded
inner scroll area retains all entries and keeps the close control reachable.
Only disclosure choices within the current page are remembered.

The public identifier scan prunes the repository-root reports directory before
walking files, so historical source snapshots cannot masquerade as public
source. A similarly named directory inside actual source remains scanned.
The release privacy gate remains unchanged: only the interface locale may be
persisted. No Studio data or browser storage is cleared by this change.

## Independent project and artifact tables

Qualification evidence, verified engine artifacts, training runs and dataset
projects now have independently scrollable table bodies. The same treatment is
used for evaluation projects, package expression profiles, local expression
drafts and TSE source projects. Existing independently scrolling Models and Jobs
surfaces are retained without another nested scrolling layer.

Table regions grow naturally for short lists and cap their height at 56% of the
viewport (up to 36 rem on desktop and 28 rem on smaller layouts). Their section
headings stay outside the scrollport and desktop column headings stick inside
it. All API records remain available. Large tables construct 40 rows first and
append the next batch near the scroll boundary; a keyboard-accessible Show more
rows control provides the same access. No filtering or backend pagination
contract was introduced. The region is keyboard-focusable and is
labelled by its existing section heading.

The qualification run count sits beside the Qualification evidence title, with
a 14 px desktop gap and 12 px compact-layout gap, instead of touching the far
edge. It still shows the existing qualification-run count; its meaning has not
changed.

Computer use on the actual port-9891 Studio verified independent scrolling with
326 qualification-evidence rows, 265 engine artifacts and 11 training projects.
The training list moved from its bottom to the top while the outer page stayed
at the same scroll position. Phone-size 390×844 and landscape 844×390 checks
verified bounded regions and no horizontal overflow; short project/profile
lists remained shorter than the maximum. These are browser viewport simulations,
not physical-phone testing.

## Selection-marker gutter and compact bundle controls

Selected rows retain the 3 px colored inset marker. The first column now has
18 px leading padding, with matching header alignment; mobile cells also keep
that gutter throughout the selected card. Long first-column names and row-select
labels wrap in full rather than being clipped or replaced with an ellipsis.

The offline bundle controls have a 960 px maximum group width on desktop.
At narrower widths the path occupies its own row and the two existing actions
share a row with equal height. The browse action, import/export handlers, path
value and validation remain unchanged. Computer use verified the 1600×900 and
390×844 layouts on an isolated synthetic fixture; no import/export operation
was submitted.

## Calm Editorial Workstation revision (2026-09-08)

This revision applies the supplied full-site redesign specification. The later
user instruction overrides its suggested expanded desktop default: the dock
starts at 72 px with SVG icons and group names; hover or keyboard-visible focus
expands an overlay to 172 px. The main workspace never changes its left edge.
Icon positions and row geometry stay fixed during the interruptible 200 ms
transition. Labels fade rather than changing between fixed and static layout.
There is no hamburger, navigation preference or new route.

Shared roles: operational titles 28 px (26 px narrow), section headings 18 px,
body/data 14 px, labels/status 12–13 px, technical data 13 px monospace.
Inspectors use a 320 px matte work surface. Collections precede inspectors when
stacked. Synthesis moves its existing language/play/stop/status nodes directly
after the editor at every width; it preserves all eight metrics, annotations,
segments, generation settings and A/B controls. Settings retain their original
field styles inside the new advanced disclosure.

Table colgroups allocate name/status/time/action columns by role rather than
uniform widths. Full raw paths use keyboard/touch-operable native technical
details; focus, open state and code scroll position are retained across polling.
Raw names are protected from status translation. Failed reads retain previous
rows and show an inline error. A first failed read does not display a zero count;
missing training/TSE progress is not rendered as zero. All changes are display
logic; job scheduling, gates and request payloads remain unchanged.

### Latest verification and limits

The detailed current evidence is in
`reports/frontend-editorial-20260908/QA.md`; screenshots, computed measurements
and machine-readable results are in `reports/frontend-ux-20260908`.
The scoped frontend acceptance is completed in the 2026-09-09 QA record below.
Native browser zoom was operated through a Docker-hosted Chromium desktop using
computer use. Mobile keyboard checks cover layout and visual-viewport simulation;
these are explicitly browser simulations, not physical-phone certification.

The Dataset Factory header now exposes the existing Target Speaker Extraction
route through a text action. This restores discoverability of the existing
controls without adding a dock icon or changing the TSE workflow.


## Final frontend acceptance — 2026-09-09

The saved QA matrix includes 129 frontend/release and naming-boundary tests, 77
module/viewport combinations, 44 zoom-equivalent reflow cases, native
125/150/200% zoom operations, and desktop/mobile computer-use workflows.
Expanded forms and dialogs use 12 px labels; their sampled text contrast passes.
The original Overview media hashes are unchanged.

Progressive tables retain selection and focus during polling and keep the full
record count visible. A 1,000-record fixture reaches every record through 24
scroll batches. On the actual Studio, 10,763 model artifacts and 319 engine
artifacts initially produced 40 rows per table. A separate 3,000-record fixture
improved median initial table readiness from 3,686 ms to 543 ms (three samples,
same Docker limits, saved pre-editorial baseline). This is a fixture comparison,
not a guarantee for every machine or backend response size. Existing API list
responses are retained; this change defers DOM creation rather than adding
server-side pagination.

Dataset refresh reads the authoritative frozen state before automatic imports,
so already-frozen datasets no longer generate repeated import attempts and
toasts. Backend immutability checks and explicit operation errors remain intact.

Expression create/edit drafts now survive polling and in-flight saves; duplicate
save submission is guarded. Mobile keyboard behavior opts into layout resizing
where supported and handles visual-viewport-only resizing without obscuring the
focused field. Large Studio select popups build their option buttons on demand;
the standalone WebUI retains its original behavior.

PWA revision editorial9 removes the old shell cache, caches only 13 static
assets, retains only the locale in browser storage, and includes the offline
document. The final live 9891 visit was offline; it was not restarted or reset.
Earlier actual-Studio checks and final isolated checks are separately recorded.
No GitHub upload or release action is part of this frontend acceptance.
