"""Probing-inspired evidence-workspace theme for the NiceGUI workbench."""

WORKBENCH_CSS = r"""
:root {
  --bp-canvas: #f4f6f8;
  --bp-panel: #ffffff;
  --bp-panel-soft: #f8fafc;
  --bp-line: #e2e8f0;
  --bp-line-strong: #cbd5e1;
  --bp-text: #111827;
  --bp-text-soft: #334155;
  --bp-muted: #64748b;
  --bp-blue: #2563eb;
  --bp-cyan: #0891b2;
  --bp-green: #15803d;
  --bp-amber: #b45309;
  --bp-red: #b91c1c;
  --bp-sidebar: #020617;
  --bp-sidebar-panel: #0f172a;
  --bp-sidebar-line: #1e293b;
  --bp-sidebar-text: #f8fafc;
  --bp-sidebar-muted: #94a3b8;
  --bp-radius: 8px;
}

html,
body,
#app,
.q-layout {
  min-height: 100%;
  color: var(--bp-text);
  background: var(--bp-canvas);
}

body {
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 14px;
  -webkit-font-smoothing: antialiased;
}

.nicegui-content {
  padding: 0;
}

*:focus-visible {
  outline: 2px solid var(--bp-blue) !important;
  outline-offset: 2px;
}

.bp-mobile-bar {
  display: none !important;
}

.bp-brand-mark {
  position: relative;
  width: 34px;
  height: 34px;
  flex: 0 0 auto;
  border: 1px solid rgba(96, 165, 250, .5);
  border-radius: 8px;
  background:
    linear-gradient(rgba(96, 165, 250, .13) 1px, transparent 1px),
    linear-gradient(90deg, rgba(96, 165, 250, .13) 1px, transparent 1px),
    #10203a;
  background-size: 8px 8px;
}

.bp-brand-mark::after {
  position: absolute;
  inset: 7px;
  content: "";
  border: 1px solid #60a5fa;
  border-top-color: transparent;
}

.bp-brand-title {
  color: var(--bp-text);
  font-size: 15px;
  font-weight: 700;
  letter-spacing: -.015em;
}

.bp-brand-subtitle {
  color: var(--bp-sidebar-muted);
  font-size: 10px;
  line-height: 1.4;
}

.bp-sidebar {
  width: 288px !important;
  color: var(--bp-sidebar-text);
  background: var(--bp-sidebar) !important;
  border-right: 1px solid var(--bp-sidebar-line) !important;
  box-shadow: none !important;
}

.bp-sidebar-shell {
  min-height: 100vh;
  gap: 0 !important;
  padding: 12px;
  overflow-y: auto;
}

.bp-sidebar-brand {
  min-height: 38px;
  padding: 0 2px;
}

.bp-sidebar .bp-brand-title {
  color: var(--bp-sidebar-text);
}

.bp-sidebar-kicker {
  color: #64748b;
  font: 650 10px ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: .13em;
}

.bp-sidebar-title {
  color: #e2e8f0;
  font-size: 13px;
  font-weight: 650;
}

.bp-mode-switch {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 4px;
  padding: 4px;
  border: 1px solid var(--bp-sidebar-line);
  border-radius: 8px;
  background: rgba(15, 23, 42, .78);
}

.bp-mode-switch .bp-mode-button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 7px;
  width: 100%;
  min-width: 0;
  min-height: 38px;
  margin: 0;
  padding: 0 6px;
  appearance: none;
  color: var(--bp-sidebar-muted);
  border: 1px solid transparent;
  border-radius: 6px;
  background: transparent;
  box-shadow: none;
  cursor: pointer;
  font-family: inherit;
  font-size: 11px;
  line-height: 1;
}

.bp-mode-switch .bp-mode-label {
  min-width: 0;
  white-space: nowrap;
}

.bp-mode-switch .bp-mode-icon {
  position: static;
  flex: 0 0 auto;
  width: 18px;
  height: 18px;
  margin: 0;
  font-size: 18px;
  line-height: 18px;
}

.bp-mode-switch .bp-mode-button--active {
  color: #dbeafe;
  border-color: rgba(96, 165, 250, .18);
  background: rgba(37, 99, 235, .18);
}

.bp-mode-switch .bp-mode-button--active .bp-mode-icon {
  color: #60a5fa;
}

.bp-mode-switch .bp-mode-button:disabled {
  cursor: default;
  opacity: .55;
}

.bp-sidebar-rule {
  margin: 11px 0;
  background: var(--bp-sidebar-line) !important;
}

.bp-sidebar-controls-card {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 8px;
  margin-top: 6px;
  padding: 10px;
  border: 1px solid var(--bp-sidebar-line);
  border-radius: 8px;
  background: rgba(15, 23, 42, .68);
}

.bp-sidebar-control {
  min-width: 0;
}

.bp-sidebar-control .q-field__control {
  min-height: 38px;
  border-radius: 6px;
  background: rgba(30, 41, 59, .86) !important;
}

.bp-sidebar-control .q-field__native,
.bp-sidebar-control .q-field__input,
.bp-sidebar-control .q-field__marginal {
  color: #e2e8f0 !important;
  font-size: 11px;
}

.bp-sidebar-control .q-field__label {
  color: #94a3b8 !important;
  font-size: 10px;
}

.bp-sidebar-field-label {
  margin: 1px 0 -2px;
  color: #64748b;
  font-size: 10px;
  font-weight: 650;
  letter-spacing: .05em;
  text-transform: uppercase;
}

.bp-sidebar-parallel-grid {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 6px;
}

.bp-sidebar-full-config {
  min-height: 32px;
  color: #bfdbfe !important;
  border-color: #334155 !important;
  font-size: 11px;
}

.bp-sidebar-primary {
  min-height: 38px;
  color: #ffffff !important;
  border-radius: 7px !important;
  background: #2563eb !important;
  font-size: 11px;
  font-weight: 650;
  box-shadow: none !important;
}

.bp-sidebar-summary {
  margin-top: 10px;
  padding: 10px;
  border: 1px solid var(--bp-sidebar-line);
  border-radius: 8px;
  background: rgba(15, 23, 42, .68);
}

.bp-sidebar-fact {
  min-width: 0;
  padding: 6px 0;
  border-top: 1px solid rgba(51, 65, 85, .58);
}

.bp-sidebar-fact-grid {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  column-gap: 10px;
}

.bp-sidebar-label {
  color: #64748b;
  font-size: 10px;
}

.bp-sidebar-value {
  width: 100%;
  overflow: hidden;
  color: #cbd5e1;
  font: 550 11px ui-monospace, SFMono-Regular, Menlo, monospace;
  line-height: 1.45;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.bp-sidebar-state {
  width: fit-content;
  margin-top: 2px;
  padding: 4px 8px;
  color: var(--bp-sidebar-muted);
  border: 1px solid #334155;
  border-radius: 999px;
  font-size: 11px;
}

.bp-sidebar-state--active {
  color: #bfdbfe;
  border-color: rgba(96, 165, 250, .35);
  background: rgba(37, 99, 235, .12);
}

.bp-sidebar-state--warning {
  color: #fde68a;
  border-color: rgba(245, 158, 11, .35);
  background: rgba(180, 83, 9, .12);
}

.bp-sidebar-state--ready {
  color: #bbf7d0;
  border-color: rgba(74, 222, 128, .3);
  background: rgba(21, 128, 61, .12);
}

.bp-sidebar-footer {
  margin-top: 12px;
  padding-top: 13px;
  border-top: 1px solid var(--bp-sidebar-line);
}

.bp-sidebar-meta {
  color: #64748b;
  font-size: 10px;
  line-height: 1.45;
}

.bp-service-dot {
  width: 6px;
  height: 6px;
  flex: 0 0 auto;
  border-radius: 50%;
  background: #22c55e;
}

.bp-sidebar-legacy {
  min-height: 34px;
  padding: 0 6px !important;
  color: var(--bp-sidebar-muted) !important;
  font-size: 11px;
}

.bp-main {
  width: 100%;
  max-width: 1600px;
  margin: 0 auto;
  padding: 20px 24px 56px;
}

.bp-workspace-heading {
  width: 100%;
  margin-bottom: 16px;
}

.bp-kicker {
  color: var(--bp-blue);
  font: 650 10px ui-monospace, SFMono-Regular, Menlo, monospace;
  letter-spacing: .1em;
  text-transform: uppercase;
}

.bp-page-title {
  margin-top: 4px;
  color: var(--bp-text);
  font-size: clamp(21px, 2vw, 27px);
  font-weight: 690;
  line-height: 1.24;
  letter-spacing: -.03em;
}

.bp-page-copy {
  max-width: 900px;
  margin-top: 4px;
  color: var(--bp-muted);
  font-size: 12px;
  line-height: 1.55;
}

.bp-setup-grid {
  width: 100%;
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
  gap: 16px;
  align-items: start;
}

.bp-focus-grid,
.bp-case-summary-grid {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 10px;
}

.bp-focus-card {
  min-width: 0;
  padding: 13px;
  border: 1px solid var(--bp-line);
  border-radius: 7px;
  background: var(--bp-panel-soft);
}

.bp-case-summary-grid {
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0;
  overflow: hidden;
  border: 1px solid var(--bp-line);
  border-radius: 7px;
  background: var(--bp-panel-soft);
}

.bp-case-summary-item {
  min-width: 0;
  padding: 11px 13px;
  border-left: 1px solid var(--bp-line);
}

.bp-case-summary-item:first-child {
  border-left: 0;
}

.bp-case-summary-value {
  overflow: hidden;
  color: var(--bp-text-soft);
  font-size: 11px;
  font-weight: 600;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.bp-setup-boundary {
  width: 100%;
  padding: 9px 11px;
  border-left: 2px solid #93c5fd;
  background: #f8fbff;
}

.bp-setup-actions {
  padding-top: 2px;
}

.bp-setup-guidance-grid {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.bp-setup-guidance {
  min-width: 0;
  padding: 2px 16px;
  border-left: 1px solid var(--bp-line);
}

.bp-setup-guidance:first-child {
  padding-left: 0;
  border-left: 0;
}

.bp-setup-guidance:last-child {
  padding-right: 0;
}

.bp-setup-config-action {
  background: #f8fbff;
}

.bp-config-root {
  width: 100%;
  gap: 0 !important;
  overflow: hidden;
  border: 1px solid var(--bp-line);
  border-radius: var(--bp-radius);
  background: var(--bp-panel);
}

.bp-config-section {
  width: 100%;
  padding: 16px;
  border: 0 !important;
  border-top: 1px solid var(--bp-line) !important;
  border-radius: 0 !important;
  background: var(--bp-panel) !important;
  box-shadow: none !important;
}

.bp-config-root > .bp-config-section:first-child {
  border-top: 0 !important;
}

.bp-config-section--search {
  background: #f8fbff !important;
}

.bp-section-index {
  min-width: 25px;
  height: 25px;
  display: grid;
  place-items: center;
  color: var(--bp-blue);
  border: 1px solid #bfdbfe;
  border-radius: 5px;
  background: #eff6ff;
  font: 650 10px ui-monospace, SFMono-Regular, Menlo, monospace;
}

.bp-section-title {
  color: var(--bp-text);
  font-size: 14px;
  font-weight: 650;
}

.bp-section-copy,
.bp-card-copy {
  color: var(--bp-muted);
  font-size: 12px;
  line-height: 1.5;
}

.bp-form-grid {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 11px;
}

.bp-form-grid--three {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.bp-form-grid > .q-field,
.bp-form-grid > .q-checkbox {
  min-width: 0;
}

.bp-config-section .q-field__control,
.bp-config-section .q-item {
  border-radius: 6px;
}

.bp-config-section .q-field--outlined .q-field__control::before {
  border-color: var(--bp-line-strong);
}

.bp-config-section .q-field--focused .q-field__control::before {
  border-color: var(--bp-blue);
}

.bp-derived-row {
  width: 100%;
  min-height: 34px;
  padding: 7px 10px;
  color: var(--bp-text-soft);
  border-left: 2px solid var(--bp-blue);
  background: #f8fafc;
  font-size: 12px;
}

.bp-advanced {
  width: 100%;
  border-top: 1px solid var(--bp-line);
}

.bp-advanced .q-item {
  min-height: 44px;
  padding: 8px 0;
}

.bp-setup-summary {
  position: sticky;
  top: 20px;
}

.bp-summary-panel,
.bp-card,
.bp-loading-panel {
  width: 100%;
  border: 1px solid var(--bp-line) !important;
  border-radius: var(--bp-radius) !important;
  background: var(--bp-panel) !important;
  box-shadow: none !important;
}

.bp-summary-panel {
  overflow: hidden;
  padding: 17px;
}

.bp-summary-rule {
  width: 100%;
  height: 1px;
  margin: 4px 0;
  background: var(--bp-line);
}

.bp-summary-label {
  color: var(--bp-muted);
  font-size: 11px;
}

.bp-summary-value {
  max-width: 205px;
  overflow: hidden;
  color: var(--bp-text-soft);
  font-size: 11px;
  text-align: right;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.bp-primary-action {
  min-height: 40px;
  border-radius: 6px !important;
  background: var(--bp-blue) !important;
  font-size: 12px;
  font-weight: 650;
  box-shadow: none !important;
}

.bp-secondary-action {
  min-height: 36px;
  border-radius: 6px !important;
  color: #1d4ed8 !important;
  border-color: #bfdbfe !important;
  background: #ffffff !important;
}

.bp-config-dialog {
  width: min(920px, calc(100vw - 48px)) !important;
  max-width: 920px !important;
  height: min(820px, calc(100vh - 48px));
  max-height: calc(100vh - 48px) !important;
  padding: 0 !important;
  overflow: hidden;
  color: var(--bp-text);
  border: 1px solid var(--bp-line) !important;
  border-radius: 10px !important;
  background: var(--bp-panel) !important;
  box-shadow: 0 24px 72px rgba(15, 23, 42, .24) !important;
}

.bp-dialog-shell {
  width: 100%;
  height: 100%;
  min-height: 0;
  gap: 0 !important;
}

.bp-dialog-heading {
  flex: 0 0 auto;
  padding: 14px 18px;
  border-bottom: 1px solid var(--bp-line);
  background: #ffffff;
}

.bp-dialog-form {
  min-height: 0;
  flex: 1 1 auto;
  align-items: stretch;
  overflow-y: auto;
  padding: 16px 18px;
  background: var(--bp-canvas);
}

.bp-dialog-footer {
  flex: 0 0 auto;
  align-items: flex-end;
  padding: 12px 18px 14px;
  border-top: 1px solid var(--bp-line);
  background: #ffffff;
}

.bp-dialog-footer .bp-primary-action {
  width: auto !important;
  min-width: 220px;
}

.bp-result-context {
  width: 100%;
  padding: 0;
}

.bp-result-context-row {
  min-height: 42px;
  flex-wrap: wrap;
}

.bp-result-chips {
  flex-wrap: wrap;
}

.bp-result-title {
  color: var(--bp-text);
  font-size: 15px;
  font-weight: 670;
  letter-spacing: -.02em;
}

.bp-context-chip,
.bp-data-chip {
  display: inline-flex;
  align-items: center;
  min-height: 22px;
  padding: 2px 7px;
  color: #475569;
  border: 1px solid var(--bp-line);
  border-radius: 5px;
  background: #ffffff;
  font: 11px ui-monospace, SFMono-Regular, Menlo, monospace;
}

.bp-result-config {
  min-height: 32px;
  padding: 0 10px !important;
}

.bp-stale-banner,
.bp-status-banner,
.bp-inline-error,
.bp-diagnostic {
  width: 100%;
  border-radius: 7px;
}

.bp-stale-banner {
  padding: 10px 12px;
  color: #92400e;
  border: 1px solid #fde68a;
  background: #fffbeb;
}

.bp-status-banner {
  padding: 10px 12px;
  color: #166534;
  border: 1px solid #bbf7d0;
  background: #f0fdf4;
}

.bp-status-banner--warning {
  color: #92400e;
  border-color: #fde68a;
  background: #fffbeb;
}

.bp-inline-error {
  padding: 10px 12px;
  color: #991b1b;
  border: 1px solid #fecaca;
  background: #fef2f2;
}

.bp-result-tabs {
  min-height: 32px;
  padding: 0 2px;
  border: 1px solid var(--bp-line);
  border-radius: 6px;
  background: #ffffff;
}

.bp-result-tabs .q-tab {
  min-height: 31px;
  padding: 0 13px;
  color: #64748b;
  font-size: 11px;
}

.bp-result-tabs .q-tab--active {
  color: var(--bp-blue);
}

.bp-result-panels,
.bp-result-panels .q-tab-panel {
  padding: 0;
  background: transparent;
}

.bp-evidence-surface {
  width: 100%;
  overflow: hidden;
  border: 1px solid var(--bp-line);
  border-radius: var(--bp-radius);
  background: var(--bp-panel);
}

.bp-evidence-section {
  width: 100%;
  display: flex;
  flex-direction: column;
  gap: 14px;
  padding: 16px;
}

.bp-evidence-section + .bp-evidence-section {
  border-top: 1px solid var(--bp-line);
}

.bp-metric-grid {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 0;
}

.bp-metric {
  min-width: 0;
  padding: 3px 16px;
  border-left: 1px solid var(--bp-line);
}

.bp-metric:first-child {
  padding-left: 0;
  border-left: 0;
}

.bp-metric-label {
  display: flex;
  align-items: center;
  gap: 6px;
  color: var(--bp-muted);
  font-size: 10px;
  font-weight: 650;
  letter-spacing: .05em;
  text-transform: uppercase;
}

.bp-metric-label::before {
  width: 5px;
  height: 5px;
  flex: 0 0 auto;
  content: "";
  border-radius: 50%;
  background: var(--metric-color, var(--bp-blue));
}

.bp-metric-value {
  margin-top: 6px;
  overflow: hidden;
  color: var(--bp-text);
  font-size: 21px;
  font-weight: 680;
  line-height: 1.2;
  letter-spacing: -.025em;
  font-variant-numeric: tabular-nums;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.bp-metric-value--range {
  font-size: 17px;
}

.bp-metric-detail {
  margin-top: 4px;
  color: var(--bp-muted);
  font-size: 11px;
}

.bp-insight {
  width: 100%;
  padding: 12px 14px;
  border-left: 3px solid var(--bp-blue);
  background: #f8fbff;
}

.bp-chain-header {
  flex-wrap: wrap;
}

.bp-fidelity-tag {
  flex: 0 0 auto;
  padding: 4px 7px;
  color: #475569;
  border: 1px solid var(--bp-line);
  border-radius: 5px;
  background: #f8fafc;
  font-size: 9px;
  letter-spacing: .05em;
}

.bp-time-treemap {
  height: 430px;
}

.bp-timeline-chart {
  height: 420px;
}

.bp-time-details {
  padding: 0 !important;
  overflow: hidden;
}

.bp-time-details > .q-expansion-item__container > .q-item {
  min-height: 42px;
  padding: 8px 12px;
}

.bp-timeline-legend {
  min-height: 24px;
}

.bp-engine-dot {
  width: 8px;
  height: 8px;
  border-radius: 2px;
  background: #64748b;
}

.bp-engine-dot--matrix {
  background: #2563eb;
}

.bp-engine-dot--vector {
  background: #7c3aed;
}

.bp-engine-dot--collective {
  background: #0891b2;
}

.bp-time-scope {
  width: 100%;
  padding: 9px 11px;
  border-left: 2px solid #93c5fd;
  background: #f8fbff;
}

.bp-time-method {
  width: 100%;
  padding: 8px 10px;
  border: 1px solid var(--bp-line);
  border-radius: 6px;
  background: #ffffff;
}

.bp-chain-stats {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 12px;
  padding-top: 10px;
  border-top: 1px solid var(--bp-line);
}

.bp-card {
  padding: 16px;
}

.bp-chart-grid,
.bp-detail-grid {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 0;
  align-items: stretch;
}

.bp-batch-filter-grid {
  width: 100%;
  display: grid;
  grid-template-columns: minmax(180px, 1.25fr) repeat(3, minmax(130px, 1fr));
  gap: 10px;
}

.bp-batch-filter {
  min-width: 0;
}

.bp-batch-filter .q-field__control {
  min-height: 40px;
  border-radius: 6px;
}

.bp-filter-clear {
  min-height: 30px;
  color: var(--bp-muted) !important;
  font-size: 11px;
}

.bp-batch-chart-grid {
  width: 100%;
  display: grid;
  grid-template-columns: minmax(0, 1.25fr) minmax(340px, .75fr);
  gap: 0;
}

.bp-batch-chart {
  height: 390px;
}

.bp-detail-grid--wide {
  grid-template-columns: minmax(0, 1.35fr) minmax(300px, .65fr);
}

.bp-evidence-block {
  min-width: 0;
  padding: 0 16px;
}

.bp-evidence-block:first-child {
  padding-left: 0;
}

.bp-evidence-block + .bp-evidence-block {
  padding-right: 0;
  border-left: 1px solid var(--bp-line);
}

.bp-card-title {
  color: var(--bp-text);
  font-size: 13px;
  font-weight: 650;
}

.bp-fact-row {
  width: 100%;
  min-height: 30px;
  padding: 5px 0;
  border-bottom: 1px solid #f1f5f9;
}

.bp-fact-row:last-child {
  border-bottom: 0;
}

.bp-stage-flow {
  width: 100%;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 11px;
}

.bp-stage {
  position: relative;
  min-width: 0;
  padding: 13px;
  border: 1px solid var(--bp-line);
  border-radius: 7px;
  background: var(--bp-panel-soft);
}

.bp-stage:not(:last-child)::after {
  position: absolute;
  top: 23px;
  right: -12px;
  z-index: 1;
  width: 12px;
  height: 1px;
  content: "";
  background: var(--bp-line-strong);
}

.bp-derivation-details {
  gap: 10px;
}

.bp-derivation-details .bp-card {
  padding: 0;
}

.bp-diagnostic {
  padding: 11px 12px;
  border: 1px solid #e5e7eb;
  border-left: 3px solid var(--diagnostic-color, var(--bp-red));
  background: #f8fafc;
}

.bp-code {
  width: 100%;
  max-height: 470px;
  overflow: auto;
  color: #e2e8f0 !important;
  border: 1px solid #1e293b;
  border-radius: 6px;
  background: #020617 !important;
  font-size: 12px;
}

.bp-grid {
  overflow: hidden;
  border: 1px solid var(--bp-line);
  border-radius: 7px;
  --ag-background-color: #ffffff;
  --ag-foreground-color: #334155;
  --ag-header-background-color: #f8fafc;
  --ag-header-foreground-color: #475569;
  --ag-border-color: #e2e8f0;
  --ag-row-border-color: #f1f5f9;
  --ag-odd-row-background-color: #fbfdff;
  --ag-selected-row-background-color: #eff6ff;
  --ag-font-size: 12px;
}

.bp-loading-panel {
  min-height: 400px;
  display: grid;
  place-items: center;
  padding: 34px;
  text-align: center;
}

.bp-loading-glyph {
  width: 56px;
  height: 56px;
  display: grid;
  place-items: center;
  border: 1px solid #bfdbfe;
  border-radius: 8px;
  background:
    linear-gradient(rgba(37, 99, 235, .08) 1px, transparent 1px),
    linear-gradient(90deg, rgba(37, 99, 235, .08) 1px, transparent 1px),
    #eff6ff;
  background-size: 10px 10px;
}

.bp-empty {
  min-height: 220px;
  display: grid;
  place-items: center;
  padding: 28px;
  text-align: center;
  border: 1px dashed var(--bp-line-strong);
  border-radius: 7px;
  background: var(--bp-panel-soft);
}

.bp-mono {
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
}

.bp-muted {
  color: var(--bp-muted);
}

.bp-positive {
  color: var(--bp-green);
}

.bp-warning {
  color: var(--bp-amber);
}

.q-expansion-item {
  border-radius: 7px;
}

.q-menu {
  color: var(--bp-text);
  background: #ffffff;
  border: 1px solid var(--bp-line);
  box-shadow: 0 12px 28px rgba(15, 23, 42, .14);
}

.bp-sidebar-menu {
  min-width: 110px !important;
  max-height: 280px !important;
  color: #e2e8f0 !important;
  border-color: #334155 !important;
  background: #0f172a !important;
  box-shadow: 0 16px 32px rgba(0, 0, 0, .34) !important;
}

.bp-sidebar-menu .q-item {
  min-height: 34px;
  color: #cbd5e1;
  font-size: 12px;
}

.bp-sidebar-menu .q-item--active,
.bp-sidebar-menu .q-item.q-manual-focusable--focused {
  color: #dbeafe;
  background: rgba(37, 99, 235, .2);
}

@media (max-width: 1180px) {
  .bp-main {
    padding: 20px 20px 48px;
  }

  .bp-setup-grid {
    grid-template-columns: 1fr;
  }

  .bp-setup-summary {
    position: static;
  }

  .bp-chart-grid,
  .bp-detail-grid,
  .bp-detail-grid--wide,
  .bp-batch-chart-grid {
    grid-template-columns: 1fr;
  }

  .bp-batch-filter-grid,
  .bp-case-summary-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }

  .bp-case-summary-item:nth-child(odd) {
    border-left: 0;
  }

  .bp-case-summary-item:nth-child(n + 3) {
    border-top: 1px solid var(--bp-line);
  }

  .bp-evidence-block {
    padding: 0 0 16px;
  }

  .bp-evidence-block + .bp-evidence-block {
    padding: 16px 0 0;
    border-top: 1px solid var(--bp-line);
    border-left: 0;
  }

  .bp-metric-grid {
    grid-template-columns: repeat(2, minmax(0, 1fr));
    row-gap: 14px;
  }

  .bp-metric:nth-child(odd) {
    padding-left: 0;
    border-left: 0;
  }

  .bp-metric:nth-child(n + 3) {
    padding-top: 14px;
    border-top: 1px solid var(--bp-line);
  }

  .bp-chain-stats {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}

@media (max-width: 980px) {
  .bp-mobile-bar {
    position: sticky;
    top: 0;
    z-index: 20;
    width: calc(100% + 40px);
    min-height: 54px;
    display: flex !important;
    margin: -20px -20px 18px;
    padding: 8px 16px;
    border-bottom: 1px solid var(--bp-line);
    background: rgba(255, 255, 255, .96);
    backdrop-filter: blur(12px);
  }

  .bp-brand-mark--mobile {
    width: 30px;
    height: 30px;
  }

  .bp-mobile-service {
    color: var(--bp-muted);
    font-size: 11px;
  }
}

@media (max-width: 900px) {
  .bp-form-grid,
  .bp-form-grid--three {
    grid-template-columns: 1fr;
  }

  .bp-setup-guidance-grid {
    grid-template-columns: 1fr;
  }

  .bp-focus-grid,
  .bp-batch-filter-grid {
    grid-template-columns: 1fr;
  }

  .bp-setup-guidance,
  .bp-setup-guidance:first-child,
  .bp-setup-guidance:last-child {
    padding: 11px 0;
    border-top: 1px solid var(--bp-line);
    border-left: 0;
  }

  .bp-setup-guidance:first-child {
    padding-top: 0;
    border-top: 0;
  }

  .bp-stage-flow {
    grid-template-columns: 1fr;
  }

  .bp-chain-stats {
    grid-template-columns: 1fr;
    gap: 6px;
  }

  .bp-stage:not(:last-child)::after {
    display: none;
  }

  .bp-config-dialog {
    width: calc(100vw - 24px) !important;
    height: calc(100vh - 24px);
    max-height: calc(100vh - 24px) !important;
  }

  .bp-dialog-heading,
  .bp-dialog-footer {
    padding-right: 13px;
    padding-left: 13px;
  }

  .bp-dialog-form {
    padding: 13px;
  }

  .bp-dialog-footer .bp-primary-action {
    width: 100% !important;
  }
}

@media (max-width: 620px) {
  .bp-main {
    padding: 18px 12px 38px;
  }

  .bp-mobile-bar {
    width: calc(100% + 24px);
    margin: -18px -12px 16px;
    padding: 8px 10px;
  }

  .bp-result-context > .q-row {
    flex-wrap: wrap;
  }

  .bp-metric-grid {
    grid-template-columns: 1fr;
    row-gap: 0;
  }

  .bp-metric,
  .bp-metric:nth-child(odd),
  .bp-metric:first-child {
    padding: 12px 0;
    border-left: 0;
  }

  .bp-metric:first-child {
    padding-top: 0;
  }

  .bp-metric:nth-child(n + 2) {
    border-top: 1px solid var(--bp-line);
  }

  .bp-result-tabs .q-tab {
    padding: 0 9px;
    font-size: 11px;
  }

  .bp-case-summary-grid {
    grid-template-columns: 1fr;
  }

  .bp-case-summary-item,
  .bp-case-summary-item:nth-child(odd) {
    border-top: 1px solid var(--bp-line);
    border-left: 0;
  }

  .bp-case-summary-item:first-child {
    border-top: 0;
  }

  .bp-setup-actions > .q-btn {
    width: 100%;
  }

  .bp-time-treemap,
  .bp-timeline-chart {
    height: 360px;
  }

  .bp-evidence-section,
  .bp-config-section,
  .bp-card {
    padding: 13px;
  }

  .bp-derivation-details .bp-card {
    padding: 0;
  }
}
"""


METRIC_COLORS = {
    "primary": "#2563eb",
    "cyan": "#0891b2",
    "violet": "#7c3aed",
    "amber": "#b45309",
    "neutral": "#64748b",
}
