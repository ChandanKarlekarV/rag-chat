# DESIGN_SYSTEM.md — rag-chat UI

Calm, readable, document-first. **No gradients anywhere** (backgrounds, buttons, text, borders, shadows). Flat color, soft neutrals, one action color plus a highlighter; study output adds soft per-topic colors on white paper. Every visual value comes from a token; no raw hex, px or font sizes inside components.

## 1. Principles
1. **Clarity over decoration.** The answer and its sources are the hero.
2. **One primary action per screen.** (Upload on empty state, Send in chat.)
3. **Always show state.** Loading, empty, error and success are designed, not afterthoughts.
4. **Trust through transparency.** Every claim shows where it came from; "I couldn't find this" looks intentional, not broken.
5. **Accessible by default.** WCAG AA contrast, keyboard reachable, visible focus, reduced-motion respected.

## 2. Color tokens

**"Notebook" palette** (redesigned 2026-11-11, replacing the earlier teal-on-dark look, which read as generic): warm paper, graphite ink, a fountain-pen blue for actions and a highlighter yellow for highlights. Dark mode is graphite, not blue-black. Soft on the eyes, high contrast for text. `app/static/css/tokens.css` is the source of truth; the tables below mirror it.

### Light
| Token | Value | Use |
|---|---|---|
| `--color-bg` | `#F5F3EE` | app background (paper) |
| `--color-surface` | `#FFFFFF` | cards, inputs, drawer |
| `--color-surface-muted` | `#EDEAE3` | code/snippet blocks, hover rows, tab tracks |
| `--color-sidebar` | `#EFECE5` | sidebar |
| `--color-border` | `#E2DED5` | dividers |
| `--color-border-strong` | `#C8C2B6` | hover borders |
| `--color-border-input` | `#8C8679` | input outlines (≥ 3:1) |
| `--color-text` | `#1C1F26` | body text (graphite) |
| `--color-text-muted` | `#5C6370` | secondary text (≥ 4.5:1 on bg) |
| `--color-text-inverse` | `#FFFFFF` | text on primary |
| `--color-primary` | `#2E4BC6` | primary buttons, links, focus ring (fountain-pen blue) |
| `--color-primary-hover` | `#233A9E` | hover/pressed |
| `--color-primary-subtle` | `#E8ECFA` | citation chips, selected rows |
| `--color-accent` | `#8A5200` | "needs attention" text |
| `--color-accent-subtle` | `#FFF1C2` | highlighter: highlighted spans and snippets |
| `--color-success` / `-subtle` | `#2B7347` / `#E3F2E8` | ready state |
| `--color-danger` / `-subtle` | `#B3261E` / `#FBE9E7` | errors, delete |
| `--color-focus` | `#2E4BC6` | 2px focus ring |

### Dark (`[data-theme="dark"]`, also via `prefers-color-scheme`)
| Token | Value |
|---|---|
| `--color-bg` | `#17181C` |
| `--color-surface` | `#1F2126` |
| `--color-surface-muted` | `#272A30` |
| `--color-sidebar` | `#1B1C21` |
| `--color-border` | `#33363D` |
| `--color-border-strong` | `#4A4E57` |
| `--color-border-input` | `#727784` |
| `--color-text` | `#ECEDEF` |
| `--color-text-muted` | `#A6ABB5` |
| `--color-primary` | `#9DB0FF` |
| `--color-primary-hover` | `#B9C7FF` |
| `--color-text-inverse` | `#111733` |
| `--color-primary-subtle` | `#262D4A` |
| `--color-accent` | `#F2C14E` |
| `--color-accent-subtle` | `#3A3018` |
| `--color-success` / `-subtle` | `#6CC48C` / `#1B3324` |
| `--color-danger` / `-subtle` | `#F08A80` / `#3D1F1C` |

### Study output: paper and topic colors (identical in both themes)
Study material is always shown on a white sheet (`.c-paper`), even in dark mode, like printed notes. Each section of the document gets its own topic color, used for its notes border, sticky notes, mind-map branch, diagrams, key terms and flashcards, with a legend at the top. Set `data-topic="1".."8"` on an element; it exposes `--t-bg`, `--t-accent` and `--t-ink`.

| Token | Value | Use |
|---|---|---|
| `--color-paper` / `-ink` / `-muted` | `#FFFFFF` / `#1C1F26` / `#5A6170` | sheet, text, secondary text |
| `--color-paper-line` / `-chip` | `#E6E2DA` / `#F2F0EB` | borders, page chips |

| Topic | bg (tint) | accent (lines) | ink (headings) |
|---|---|---|---|
| 1 blue | `#EAF0FC` | `#3D6BD6` | `#1F4290` |
| 2 green | `#E6F4EA` | `#2E9455` | `#1B6136` |
| 3 ochre | `#FDF2D8` | `#B07A0C` | `#6E4800` |
| 4 rose | `#FCE9EC` | `#D04A66` | `#962A42` |
| 5 violet | `#EFEAFB` | `#7A5BC7` | `#4C3391` |
| 6 teal | `#E2F3F1` | `#24918A` | `#145E59` |
| 7 orange | `#FDECDF` | `#D4682A` | `#8F3E0C` |
| 8 slate | `#ECEFF3` | `#66768A` | `#364352` |

**Contrast rule:** body text ≥ 4.5:1, large text and UI borders ≥ 3:1. `scripts/check_contrast.py` reads the token values and fails if any listed pair is under the threshold, in both themes: text/bg, text-muted/bg and /sidebar, text-inverse/primary, primary/surface, accent/accent-subtle, danger/danger-subtle, and for the paper: ink and muted on paper, ink and paper-ink on every topic tint, white on every topic ink, every topic accent on paper (3:1). Adjust token values, not the rule.

## 3. Typography
Font: `Inter`, falling back to `system-ui, -apple-system, "Segoe UI", sans-serif`. Mono: `ui-monospace, "SF Mono", Menlo, monospace`.

| Token | Size / line-height | Weight | Use |
|---|---|---|---|
| `--text-xs` | 12 / 16 | 500 | badges, meta |
| `--text-sm` | 14 / 20 | 400–500 | secondary UI, sidebar |
| `--text-base` | 16 / 24 | 400 | chat answers, inputs |
| `--text-lg` | 18 / 28 | 600 | section titles |
| `--text-xl` | 24 / 32 | 600 | page title / empty state |

Answers use `--text-base` with max line length ~72ch. Letter-spacing normal. Never use ALL CAPS for body UI.

## 4. Spacing (4 px base)
`--space-1: 4px · -2: 8px · -3: 12px · -4: 16px · -5: 20px · -6: 24px · -8: 32px · -10: 40px · -12: 48px`

Rules:
- Only these steps. No odd values (no 5px, 7px, 13px).
- **Inside a component:** `space-2` to `space-4`. **Between related components:** `space-3` to `space-4`. **Between sections:** `space-6` to `space-8`.
- Page gutter: `space-4` on mobile, `space-6` on desktop. Sidebar width 320px (desktop), full-screen sheet on mobile.
- Minimum tap target 40×40px (44px on touch devices).
- Align to the grid: siblings share the same left edge; no free-floating offsets.

## 5. Shape, borders, elevation (flat)
- Radius: `--radius-sm: 6px` (chips, badges), `--radius-md: 10px` (inputs, buttons, cards), `--radius-lg: 14px` (drawer, modal), `--radius-full: 999px` (avatars, pills).
- Border: 1px `--color-border`. Cards rely on border + subtle shadow, not heavy shadow.
- Shadows (solid, low-opacity, single layer): `--shadow-sm: 0 1px 2px rgba(31,41,51,.06)`, `--shadow-md: 0 4px 12px rgba(31,41,51,.08)`. Used only on drawer, modal, toast, dropdown.
- **Forbidden:** `linear-gradient`, `radial-gradient`, `conic-gradient`, `background-clip:text`, glassmorphism/backdrop blur, neon glows.

## 6. Motion
`--duration-fast: 120ms`, `--duration-base: 200ms`, easing `cubic-bezier(.2,.0,.0,1)`. Animate opacity/transform only. Streaming text appears token by token without extra animation. Honor `prefers-reduced-motion` (disable transitions and skeleton shimmer; use static placeholders).

## 7. Token implementation

Define tokens once as CSS custom properties in `:root` / `[data-theme="dark"]`, and map them into the Tailwind CDN config so utilities read from tokens:

```html
<script>
  tailwind.config = {
    theme: { extend: {
      colors: {
        bg: 'var(--color-bg)', surface: 'var(--color-surface)',
        muted: 'var(--color-surface-muted)', border: 'var(--color-border)',
        ink: 'var(--color-text)', 'ink-muted': 'var(--color-text-muted)',
        primary: 'var(--color-primary)', 'primary-hover': 'var(--color-primary-hover)',
        'primary-subtle': 'var(--color-primary-subtle)',
        accent: 'var(--color-accent)', 'accent-subtle': 'var(--color-accent-subtle)',
        danger: 'var(--color-danger)', success: 'var(--color-success)'
      },
      borderRadius: { sm: '6px', md: '10px', lg: '14px' },
      fontFamily: { sans: ['Inter','system-ui','sans-serif'] }
    }}
  }
</script>
```
(Verify the current Tailwind CDN configuration syntax in the Tailwind docs via Context7 before writing it; Tailwind's config approach has changed between major versions.)

## 8. Component conventions

**Naming:** components are classes prefixed `c-` (`c-btn`, `c-card`), variants `c-btn--primary`, state via `data-state` or `aria-*` (`aria-busy`, `aria-selected`, `aria-invalid`). JS hooks use `data-js="..."`, never styling classes.
**Where:** one `components.css` section per component, each built only from tokens. Utility classes in HTML only for layout (flex, grid, gap, padding from the spacing scale).
**Every interactive component defines:** default, hover, active, focus-visible (2px ring `--color-focus`, 2px offset), disabled, loading.
**Content rules:** sentence case, plain words, verbs on buttons ("Upload PDF", "Ask"), no jargon in user-facing text.

| Component | Spec |
|---|---|
| **Button** | Variants: primary (solid teal, inverse text), secondary (surface + border), ghost (text only), danger. Sizes: md 40px, sm 32px. Loading shows a spinner and `aria-busy`. One primary per view. |
| **Input / Textarea** | 40px min, 1px border, radius-md, label above (never placeholder-only), helper text `--text-sm` muted, error text danger + `aria-invalid` + message tied by `aria-describedby`. |
| **Card** | surface, 1px border, radius-md, padding `space-4`. |
| **Dropzone** | dashed 1.5px border, muted surface; hover/drag-over switches to primary-subtle + primary border; text "Drop a PDF here or choose a file"; shows limits "PDF, up to 20 MB". |
| **Document row** | icon + filename (truncate) + meta (pages · date) + status badge + checkbox to include in questions + overflow menu (Delete). Selected = primary-subtle. |
| **Status badge** | pill, `--text-xs`; Processing (muted + spinner), Ready (success-subtle), Failed (danger-subtle) with a retry link. Never color alone: always an icon or text. |
| **Chat message** | User: right-aligned, primary-subtle bubble. Assistant: left, no bubble (surface card), citations row beneath, actions (copy, retry) appearing on hover/focus. Max width 72ch. |
| **Citation chip** | `[1]` inline = small chip, primary-subtle bg, primary text, radius-sm, keyboard focusable; opens the source drawer. |
| **Source drawer** | right side (desktop) / bottom sheet (mobile); filename, page range, snippet on accent-subtle with the matching text emphasized, "Open page N" optional. Esc closes, focus is trapped, focus returns to the chip. |
| **Fallback message** | Neutral card (not an error): "I couldn't find this in your documents." + 2 suggestions: rephrase, or upload a relevant file. |
| **Toast** | bottom center, auto-dismiss 5s, `role="status"`; errors persist until dismissed. |
| **Skeleton** | muted blocks, no shimmer gradient (use opacity pulse; static when reduced motion). |
| **Empty state** | Icon, title, one sentence, primary action, and 3 clickable sample questions once a document is ready. |
| **Modal** | only for destructive confirms; focus trapped; default focus on Cancel. |

## 9. Layout
Desktop: 2 columns — sidebar (documents) 320px + main chat (flex). Chat input sticky at the bottom with `space-4` padding; messages scroll. Mobile (<768px): sidebar becomes a slide-over opened by a "Documents" button; drawer becomes a bottom sheet. Breakpoints: 640 / 768 / 1024.

## 10. User-friendliness requirements (checklist)
- [ ] First run: three-step hint (1 Upload · 2 Ask · 3 Check sources) and a **"Try with a sample PDF / continue as guest"** button.
- [ ] Upload shows real stages: Reading → Splitting → Understanding → Ready, with percent where possible.
- [ ] Questions can be asked while other docs process; the input explains when no document is ready ("Upload a PDF to start").
- [ ] Enter = send, Shift+Enter = newline, Esc closes drawer, `/` focuses the input; shortcuts listed in a small "?" popover.
- [ ] Stop-generating button during streaming; Retry on failures; Copy on answers.
- [ ] Sample questions generated per document (first run: 3 static suggestions).
- [ ] Conversation history in the sidebar tab with rename/delete.
- [ ] Clear scope indicator above the input: "Searching 2 of 3 documents".
- [ ] Friendly errors with a next step; never show stack traces or raw status codes.
- [ ] Streaming region is `aria-live="polite"`; all icons have text alternatives; color is never the only signal.
- [ ] Light/dark follows the OS, with a manual toggle remembered per user.
- [ ] Works at 360px width without horizontal scroll.

## 11. Definition of done for the UI
Zero gradients (grep for `gradient` must return nothing in `static/`); no hard-coded colors outside the token block (grep for `#` hex only in the token section); contrast script passes in both themes; all components in section 8 exist and are demonstrated on a `/static/styleguide.html` page.
