/**
 * Contribute form — main entry point.
 *
 * Multi-step wizard:
 *   Step 1: City Identity
 *   Step 2: Data Source (with ArcGIS probing)
 *   Step 3: Field Mapping & Geographic Scope
 *   Step 4: Scale & Extras
 *   Step 5: Review & Submit
 *
 * Form state is persisted in localStorage so users can refresh without losing progress.
 */

import {
  DEFAULT_STATE,
  deriveCityKey,
  generateMarkdown,
  suggestPmtiles,
} from './template';
import type { ContributeFormState } from './template';
import {
  probeArcGISLayer,
  looksLikeArcGISUrl,
  normalizeArcGISUrl,
  suggestFields,
  describeCRS,
  describeGeometryType,
} from './arcgis';
import type { ArcGISLayerInfo } from './arcgis';
import {
  downloadMarkdown,
  buildGitHubNewIssueUrl,
  getNewIssueUrl,
} from './github';

const STORAGE_KEY = 'civicmapper_contribute_state';
const TOTAL_STEPS = 5;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let state: ContributeFormState = loadState();
let currentStep = 1;
let arcgisInfo: ArcGISLayerInfo | null = null;

function loadState(): ContributeFormState {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return { ...DEFAULT_STATE, ...JSON.parse(raw) };
  } catch { /* ignore */ }
  return { ...DEFAULT_STATE };
}

function saveState() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  } catch { /* ignore */ }
}

function resetState() {
  state = { ...DEFAULT_STATE };
  localStorage.removeItem(STORAGE_KEY);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function el<T extends HTMLElement = HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

function val(id: string): string {
  return (el<HTMLInputElement>(id)?.value ?? '').trim();
}

function checked(id: string): boolean {
  return (el<HTMLInputElement>(id)?.checked ?? false);
}

function setVal(id: string, value: string | boolean) {
  const element = el<HTMLInputElement>(id);
  if (!element) return;
  if (typeof value === 'boolean') {
    element.checked = value;
  } else {
    element.value = value;
  }
}

function setText(id: string, text: string) {
  const e = el(id);
  if (e) e.textContent = text;
}

function setStatus(id: string, msg: string, type: 'info' | 'success' | 'error' | '') {
  const e = el(id);
  if (!e) return;
  e.textContent = msg;
  e.className = `status-msg ${type}`;
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------

function showStep(step: number) {
  for (let i = 1; i <= TOTAL_STEPS; i++) {
    const stepEl = el(`step-${i}`);
    if (stepEl) stepEl.style.display = i === step ? '' : 'none';
  }
  currentStep = step;
  updateProgressBar();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

function updateProgressBar() {
  for (let i = 1; i <= TOTAL_STEPS; i++) {
    const dot = el(`progress-${i}`);
    if (!dot) continue;
    dot.classList.toggle('active', i === currentStep);
    dot.classList.toggle('done', i < currentStep);
  }
  setText('step-label', `Step ${currentStep} of ${TOTAL_STEPS}`);
}

function goNext() {
  if (!validateCurrentStep()) return;
  collectCurrentStep();
  saveState();
  if (currentStep < TOTAL_STEPS) {
    if (currentStep === 4) buildReview();
    showStep(currentStep + 1);
  }
}

function goPrev() {
  if (currentStep > 1) showStep(currentStep - 1);
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

function validateCurrentStep(): boolean {
  switch (currentStep) {
    case 1:
      if (!val('f-display-name')) {
        alert('Please enter the city display name.');
        return false;
      }
      if (!val('f-state-code') || val('f-state-code').length !== 2) {
        alert('Please enter a valid 2-letter state code.');
        return false;
      }
      if (!val('f-city-key')) {
        alert('City key is required.');
        return false;
      }
      return true;
    case 2:
      if (!val('f-source-url')) {
        alert('Please enter the data source URL.');
        return false;
      }
      return true;
    case 3:
      if (!val('f-land-value-field')) {
        alert('Land value field is required.');
        return false;
      }
      if (!val('f-land-use-field')) {
        alert('Land use category field is required.');
        return false;
      }
      return true;
    default:
      return true;
  }
}

// ---------------------------------------------------------------------------
// Collect form values into state
// ---------------------------------------------------------------------------

function collectStep1() {
  state.displayName = val('f-display-name');
  state.cityKey = val('f-city-key');
  state.stateCode = val('f-state-code').toLowerCase();
}

function collectStep2() {
  state.sourceUrl = val('f-source-url');
  state.sourceType = val('f-source-type');
  state.layerPath = arcgisInfo ? arcgisInfo.name : val('f-layer-path');
  state.authentication = val('f-authentication');
  state.updateFrequency = val('f-update-frequency');
  state.dataVintage = val('f-data-vintage');
  state.sourceCRS = arcgisInfo?.spatialReference?.latestWkid
    ? describeCRS(arcgisInfo.spatialReference.latestWkid)
    : val('f-source-crs');
  state.geometryIncluded = val('f-geometry-included');
}

function collectStep3() {
  state.parcelIdField = val('f-parcel-id-field');
  state.landValueField = val('f-land-value-field');
  state.improvementValueField = val('f-improvement-value-field');
  state.landUseCategoryField = val('f-land-use-field');
  state.ownerNameField = val('f-owner-name-field') || 'not used';
  state.sourceCoverage = val('f-source-coverage');
  state.clipMethod = val('f-clip-method');
  state.filterField = val('f-filter-field');
  state.boundarySource = val('f-boundary-source');
  state.geographyNotes = val('f-geography-notes');
  state.assessorLinkAvailable = checked('f-assessor-link-available');
  state.assessorLinkPattern = val('f-assessor-link-pattern');
  state.assessorLinkIdField = val('f-assessor-link-id-field');
}

function collectStep4() {
  state.approxParcelCount = val('f-approx-parcel-count');
  state.pmtilesRecommended = checked('f-pmtiles-recommended');
  state.includeParking = checked('f-include-parking');
  state.contributorNotes = val('f-contributor-notes');
}

function collectCurrentStep() {
  switch (currentStep) {
    case 1: collectStep1(); break;
    case 2: collectStep2(); break;
    case 3: collectStep3(); break;
    case 4: collectStep4(); break;
  }
}

// ---------------------------------------------------------------------------
// Step 2 — ArcGIS probing
// ---------------------------------------------------------------------------

async function probeUrl() {
  const url = val('f-source-url');
  if (!url) return;

  const normalized = normalizeArcGISUrl(url);
  setVal('f-source-url', normalized);
  setStatus('probe-status', 'Probing data source…', 'info');

  el('btn-probe')?.setAttribute('disabled', 'true');

  if (!looksLikeArcGISUrl(normalized)) {
    setStatus('probe-status', 'URL does not look like an ArcGIS FeatureServer — fill in fields manually below.', 'info');
    el('btn-probe')?.removeAttribute('disabled');
    return;
  }

  const info = await probeArcGISLayer(normalized);
  el('btn-probe')?.removeAttribute('disabled');

  if (!info) {
    setStatus('probe-status', 'Could not connect (CORS or invalid URL). Please fill in the fields manually.', 'error');
    return;
  }

  arcgisInfo = info;
  setStatus('probe-status', `✅ Connected: "${info.name}" — ${info.fields.length} fields, geometry: ${describeGeometryType(info.geometryType)}`, 'success');

  // Pre-fill layer path
  setVal('f-layer-path', info.name);

  // Pre-fill CRS
  const wkid = info.spatialReference?.latestWkid ?? info.spatialReference?.wkid;
  if (wkid) setVal('f-source-crs', describeCRS(wkid));

  // Pre-fill geometry type
  if (info.approxCount) {
    setVal('f-approx-parcel-count', `~${Math.round(info.approxCount / 1000) * 1000}`);
    const pmtiles = suggestPmtiles(String(info.approxCount));
    setVal('f-pmtiles-recommended', pmtiles);
  }

  // Populate field selects in step 3
  populateFieldSelects(info);

  // Suggest field mappings
  const suggestions = suggestFields(info.fields);
  if (suggestions.parcelId) setVal('f-parcel-id-field', suggestions.parcelId);
  if (suggestions.landValue) setVal('f-land-value-field', suggestions.landValue);
  if (suggestions.improvementValue) setVal('f-improvement-value-field', suggestions.improvementValue);
  if (suggestions.landUseCategory) setVal('f-land-use-field', suggestions.landUseCategory);
}

function populateFieldSelects(info: ArcGISLayerInfo) {
  const selectIds = ['f-parcel-id-field', 'f-land-value-field', 'f-improvement-value-field', 'f-land-use-field', 'f-owner-name-field'];
  for (const id of selectIds) {
    const sel = el<HTMLSelectElement>(id);
    if (!sel) continue;
    // Change to a select element (if it was an input, it stays as-is for simplicity)
    // We populate a datalist instead
  }

  // Populate datalists
  for (const listId of ['dl-parcel', 'dl-value', 'dl-impr', 'dl-lu', 'dl-owner']) {
    const dl = el(listId);
    if (!dl) continue;
    dl.innerHTML = '';
    for (const f of info.fields) {
      const opt = document.createElement('option');
      opt.value = f.name;
      opt.label = f.alias !== f.name ? `${f.alias} (${f.name})` : f.name;
      dl.appendChild(opt);
    }
  }
}

// ---------------------------------------------------------------------------
// Step 4 — Review
// ---------------------------------------------------------------------------

function buildReview() {
  collectStep4();
  const md = generateMarkdown(state);
  const preview = el('preview-markdown');
  if (preview) preview.textContent = md;

  setText('review-city', `${state.displayName}, ${state.stateCode.toUpperCase()}`);
}

function doDownload() {
  collectStep4();
  const md = generateMarkdown(state);
  const filename = `${state.cityKey}.md`;
  downloadMarkdown(filename, md);
}

function doOpenGitHub() {
  collectStep4();
  const md = generateMarkdown(state);
  const url = buildGitHubNewIssueUrl({
    stateCode: state.stateCode,
    cityKey: state.cityKey,
    markdown: md,
  });

  if (url) {
    window.open(url, '_blank');
  } else {
    // Proposal too large for URL pre-fill — download it, open a blank issue to paste into
    doDownload();
    window.open(getNewIssueUrl(), '_blank');
    el('gh-prefill-note')?.style && (el('gh-prefill-note')!.style.display = '');
  }
}

// ---------------------------------------------------------------------------
// Restore form from saved state
// ---------------------------------------------------------------------------

function restoreForm() {
  // Step 1
  setVal('f-display-name', state.displayName);
  setVal('f-city-key', state.cityKey);
  setVal('f-state-code', state.stateCode);
  // Step 2
  setVal('f-source-url', state.sourceUrl);
  setVal('f-source-type', state.sourceType);
  setVal('f-layer-path', state.layerPath);
  setVal('f-authentication', state.authentication);
  setVal('f-update-frequency', state.updateFrequency);
  setVal('f-data-vintage', state.dataVintage);
  setVal('f-source-crs', state.sourceCRS);
  setVal('f-geometry-included', state.geometryIncluded);
  // Step 3
  setVal('f-parcel-id-field', state.parcelIdField);
  setVal('f-land-value-field', state.landValueField);
  setVal('f-improvement-value-field', state.improvementValueField);
  setVal('f-land-use-field', state.landUseCategoryField);
  setVal('f-owner-name-field', state.ownerNameField === 'not used' ? '' : state.ownerNameField);
  setVal('f-source-coverage', state.sourceCoverage);
  setVal('f-clip-method', state.clipMethod);
  setVal('f-filter-field', state.filterField);
  setVal('f-boundary-source', state.boundarySource);
  setVal('f-geography-notes', state.geographyNotes);
  setVal('f-assessor-link-available', state.assessorLinkAvailable);
  setVal('f-assessor-link-pattern', state.assessorLinkPattern);
  setVal('f-assessor-link-id-field', state.assessorLinkIdField);
  // Step 4
  setVal('f-approx-parcel-count', state.approxParcelCount);
  setVal('f-pmtiles-recommended', state.pmtilesRecommended);
  setVal('f-include-parking', state.includeParking);
  setVal('f-contributor-notes', state.contributorNotes);
}

// ---------------------------------------------------------------------------
// Wire up auto-derive of city key from display name
// ---------------------------------------------------------------------------

function setupAutoDerive() {
  const nameInput = el<HTMLInputElement>('f-display-name');
  const keyInput = el<HTMLInputElement>('f-city-key');
  if (!nameInput || !keyInput) return;

  nameInput.addEventListener('input', () => {
    if (!keyInput.dataset.manual) {
      keyInput.value = deriveCityKey(nameInput.value);
    }
  });

  keyInput.addEventListener('input', () => {
    keyInput.dataset.manual = '1';
  });

  // Show/hide clip fields based on coverage
  const coverageSelect = el<HTMLSelectElement>('f-source-coverage');
  const clipFields = el('clip-fields');
  if (coverageSelect && clipFields) {
    coverageSelect.addEventListener('change', () => {
      clipFields.style.display = coverageSelect.value.startsWith('County') ? '' : 'none';
    });
  }

  // Toggle assessor link fields
  const assessorCheck = el<HTMLInputElement>('f-assessor-link-available');
  const assessorFields = el('assessor-link-fields');
  if (assessorCheck && assessorFields) {
    assessorCheck.addEventListener('change', () => {
      assessorFields.style.display = assessorCheck.checked ? '' : 'none';
    });
  }
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  restoreForm();
  showStep(1);
  setupAutoDerive();

  el('btn-next-1')?.addEventListener('click', goNext);
  el('btn-next-2')?.addEventListener('click', goNext);
  el('btn-next-3')?.addEventListener('click', goNext);
  el('btn-next-4')?.addEventListener('click', goNext);

  el('btn-prev-2')?.addEventListener('click', goPrev);
  el('btn-prev-3')?.addEventListener('click', goPrev);
  el('btn-prev-4')?.addEventListener('click', goPrev);
  el('btn-prev-5')?.addEventListener('click', goPrev);

  el('btn-probe')?.addEventListener('click', probeUrl);

  el('btn-download')?.addEventListener('click', doDownload);
  el('btn-github')?.addEventListener('click', doOpenGitHub);

  el('btn-reset')?.addEventListener('click', () => {
    if (confirm('Reset the form? All progress will be lost.')) {
      resetState();
      restoreForm();
      showStep(1);
    }
  });
});
