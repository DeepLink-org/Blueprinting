"""Blueprinting visual tokens for the NiceGUI workbench."""

WORKBENCH_CSS = r"""
:root {
  --bp-bg: #070a12;
  --bp-surface: rgba(15, 20, 34, .82);
  --bp-surface-strong: rgba(20, 27, 45, .96);
  --bp-border: rgba(148, 163, 184, .14);
  --bp-border-bright: rgba(106, 119, 255, .42);
  --bp-text: #edf2ff;
  --bp-muted: #8d9ab4;
  --bp-primary: #7c68ff;
  --bp-cyan: #22d3ee;
  --bp-violet: #a78bfa;
  --bp-amber: #f59e0b;
  --bp-danger: #fb7185;
  --bp-success: #34d399;
  --bp-radius: 18px;
  --bp-shadow: 0 22px 70px rgba(0, 0, 0, .28);
}

html, body, #app, .q-layout {
  min-height: 100%;
  color: var(--bp-text);
  background:
    radial-gradient(circle at 74% -10%, rgba(88, 77, 255, .18), transparent 36%),
    radial-gradient(circle at 12% 38%, rgba(34, 211, 238, .07), transparent 28%),
    var(--bp-bg);
}

body {
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.nicegui-content {
  padding: 0;
}

.bp-header {
  min-height: 70px;
  padding: 0 22px;
  background: rgba(7, 10, 18, .78) !important;
  border-bottom: 1px solid var(--bp-border);
  backdrop-filter: blur(22px);
}

.bp-brand-mark {
  width: 38px;
  height: 38px;
  display: grid;
  place-items: center;
  border: 1px solid rgba(124, 104, 255, .58);
  border-radius: 12px;
  background: linear-gradient(145deg, rgba(124, 104, 255, .32), rgba(34, 211, 238, .08));
  box-shadow: 0 0 30px rgba(124, 104, 255, .22), inset 0 1px 0 rgba(255, 255, 255, .12);
}

.bp-brand-title {
  font-size: 17px;
  font-weight: 720;
  letter-spacing: -.02em;
}

.bp-brand-subtitle {
  color: var(--bp-muted);
  font-size: 10px;
  letter-spacing: .12em;
  text-transform: uppercase;
}

.bp-header .q-tab {
  min-height: 68px;
  padding: 0 15px;
  color: #8290aa;
  font-size: 12px;
  letter-spacing: .01em;
}

.bp-header .q-tab--active {
  color: #f5f7ff;
}

.bp-drawer {
  width: 318px !important;
  padding: 16px 16px 28px;
  background: rgba(9, 13, 23, .94) !important;
  border-right: 1px solid var(--bp-border) !important;
  backdrop-filter: blur(24px);
}

.bp-drawer-title {
  color: #dfe7fa;
  font-size: 12px;
  font-weight: 700;
  letter-spacing: .1em;
  text-transform: uppercase;
}

.bp-form-section {
  width: 100%;
  padding: 12px;
  border: 1px solid var(--bp-border);
  border-radius: 14px;
  background: rgba(17, 23, 39, .52);
}

.bp-form-section .q-field__control,
.bp-form-section .q-item {
  border-radius: 10px;
}

.bp-main {
  width: 100%;
  max-width: 1540px;
  margin: 0 auto;
  padding: 28px 32px 60px;
}

.bp-hero {
  position: relative;
  width: 100%;
  min-height: 164px;
  overflow: hidden;
  padding: 28px 30px;
  border: 1px solid var(--bp-border-bright);
  border-radius: 24px;
  background:
    linear-gradient(110deg, rgba(35, 31, 74, .96), rgba(15, 23, 42, .86) 55%, rgba(8, 31, 42, .82)),
    var(--bp-surface);
  box-shadow: var(--bp-shadow), inset 0 1px 0 rgba(255, 255, 255, .08);
}

.bp-hero::after {
  content: "";
  position: absolute;
  right: -65px;
  top: -100px;
  width: 330px;
  height: 330px;
  border: 1px solid rgba(34, 211, 238, .18);
  border-radius: 50%;
  box-shadow: 0 0 0 36px rgba(124, 104, 255, .045), 0 0 0 82px rgba(34, 211, 238, .028);
}

.bp-kicker {
  color: #7ee8f7;
  font-size: 11px;
  font-weight: 750;
  letter-spacing: .16em;
  text-transform: uppercase;
}

.bp-hero-title {
  max-width: 850px;
  margin-top: 7px;
  font-size: clamp(24px, 3vw, 38px);
  font-weight: 740;
  line-height: 1.12;
  letter-spacing: -.035em;
}

.bp-hero-copy {
  max-width: 850px;
  margin-top: 10px;
  color: #aab7d0;
  font-size: 13px;
  line-height: 1.7;
}

.bp-chip {
  padding: 6px 10px;
  color: #b9c4dc;
  font: 600 10px ui-monospace, SFMono-Regular, Menlo, monospace;
  border: 1px solid rgba(148, 163, 184, .16);
  border-radius: 999px;
  background: rgba(5, 8, 15, .35);
}

.bp-section-title {
  margin-top: 6px;
  color: #e8edfb;
  font-size: 17px;
  font-weight: 690;
  letter-spacing: -.018em;
}

.bp-section-copy {
  max-width: 900px;
  color: var(--bp-muted);
  font-size: 12px;
  line-height: 1.7;
}

.bp-card {
  width: 100%;
  border: 1px solid var(--bp-border) !important;
  border-radius: var(--bp-radius) !important;
  background: var(--bp-surface) !important;
  box-shadow: 0 12px 45px rgba(0, 0, 0, .14) !important;
  backdrop-filter: blur(16px);
}

.bp-metric {
  position: relative;
  min-width: 170px;
  flex: 1 1 170px;
  padding: 17px 18px;
  overflow: hidden;
  border: 1px solid var(--bp-border);
  border-radius: 16px;
  background: linear-gradient(160deg, rgba(24, 31, 51, .92), rgba(12, 17, 29, .92));
}

.bp-metric::before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  width: 3px;
  height: 100%;
  background: var(--metric-color, #7c68ff);
  box-shadow: 0 0 18px var(--metric-color, #7c68ff);
}

.bp-metric-label {
  color: var(--bp-muted);
  font-size: 10px;
  font-weight: 650;
  letter-spacing: .08em;
  text-transform: uppercase;
}

.bp-metric-value {
  margin-top: 7px;
  color: #f4f7ff;
  font: 680 23px ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: -.04em;
}

.bp-metric-detail {
  margin-top: 4px;
  color: #73809a;
  font-size: 10px;
}

.bp-status {
  width: 100%;
  padding: 11px 14px;
  border: 1px solid rgba(52, 211, 153, .24);
  border-radius: 12px;
  color: #a7f3d0;
  background: rgba(16, 185, 129, .07);
}

.bp-status--warning {
  color: #fde68a;
  border-color: rgba(245, 158, 11, .28);
  background: rgba(245, 158, 11, .08);
}

.bp-diagnostic {
  width: 100%;
  padding: 12px 14px;
  border-left: 3px solid var(--diagnostic-color, #7c68ff);
  border-radius: 10px;
  background: rgba(20, 27, 45, .74);
}

.bp-code {
  width: 100%;
  max-height: 470px;
  overflow: auto;
  border: 1px solid var(--bp-border);
  border-radius: 12px;
  background: #060911 !important;
}

.bp-stage {
  min-width: 210px;
  flex: 1 1 210px;
  padding: 15px;
  border: 1px solid var(--bp-border);
  border-radius: 14px;
  background: rgba(16, 22, 37, .76);
}

.bp-stage-valid {
  color: var(--bp-success);
}

.bp-empty {
  min-height: 240px;
  display: grid;
  place-items: center;
  text-align: center;
  border: 1px dashed rgba(148, 163, 184, .2);
  border-radius: var(--bp-radius);
  background: rgba(15, 20, 34, .42);
}

.bp-grid {
  overflow: hidden;
  border: 1px solid var(--bp-border);
  border-radius: 14px;
}

.bp-mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.bp-muted {
  color: var(--bp-muted);
}

.q-tab-panel {
  padding: 22px 0 0;
  background: transparent;
}

.q-expansion-item {
  border-radius: 12px;
}

@media (max-width: 900px) {
  .bp-main { padding: 18px 16px 44px; }
  .bp-header { padding: 0 10px; }
  .bp-brand-subtitle { display: none; }
  .bp-header .q-tab { padding: 0 7px; font-size: 10px; }
  .bp-hero { padding: 22px 20px; }
}
"""


METRIC_COLORS = {
    "primary": "#7c68ff",
    "cyan": "#22d3ee",
    "violet": "#a78bfa",
    "amber": "#f59e0b",
    "neutral": "#64748b",
}
