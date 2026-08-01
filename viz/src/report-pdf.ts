/**
 * Branded PDF city report — client-side via jsPDF (lazy-loaded on click).
 *
 * Walks the app's three views (3D land value, Underused, Parking), snapshots
 * each map canvas (maps are created with preserveDrawingBuffer: true), scrapes
 * the already-formatted stats from the sidebar DOM, and lays everything out in
 * a CLE-branded A4 document with a link back to the live map. The report is
 * itself a funnel artifact: every page footer carries civicmapper.org,
 * landeconomics.org, and the Substack.
 */

import type maplibregl from 'maplibre-gl';

export type ReportContext = {
  map: maplibregl.Map;
  mapUnder: maplibregl.Map;
  getParkingMap: () => maplibregl.Map | null;
  setTab: (tab: 'main' | 'under' | 'parking') => void;
  getTab: () => string;
  cityLabel: () => string;
  tabsAvailable: () => { under: boolean; parking: boolean };
};

const PURPLE: [number, number, number] = [52, 40, 119];   // #342877
const DARK: [number, number, number] = [23, 24, 28];
const GRAY: [number, number, number] = [91, 94, 107];
const LIGHT_BG: [number, number, number] = [241, 239, 249]; // primary-50

const PAGE_W = 210;
const PAGE_H = 297;
const MARGIN = 16;
const CONTENT_W = PAGE_W - MARGIN * 2;

function waitForIdle(map: maplibregl.Map, timeoutMs: number): Promise<void> {
  return new Promise((resolve) => {
    let done = false;
    const finish = () => { if (!done) { done = true; resolve(); } };
    map.once('idle', finish);
    map.triggerRepaint();
    window.setTimeout(finish, timeoutMs);
  });
}

// Snapshot the canvas immediately after a frame is drawn: request a repaint
// and read the canvas inside the `render` event, so the back buffer is never
// stale/black (e.g. right after the map's tab was unhidden).
function captureMap(map: maplibregl.Map, timeoutMs = 4000): Promise<{ data: string; w: number; h: number } | null> {
  return new Promise((resolve) => {
    let settled = false;
    const finish = () => {
      if (settled) return;
      settled = true;
      try {
        const canvas = map.getCanvas();
        if (!canvas.width || !canvas.height) return resolve(null);
        resolve({ data: canvas.toDataURL('image/jpeg', 0.85), w: canvas.width, h: canvas.height });
      } catch {
        resolve(null);
      }
    };
    map.once('render', finish);
    map.triggerRepaint();
    window.setTimeout(finish, timeoutMs);
  });
}

async function loadImage(url: string): Promise<{ data: string; w: number; h: number } | null> {
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext('2d')!.drawImage(img, 0, 0);
    return { data: canvas.toDataURL('image/png'), w: canvas.width, h: canvas.height };
  } catch {
    return null;
  }
}

type UnderStat = { label: string; value: string; meta: string; buckets: { label: string; value: string }[] };

function scrapeUnderStats(): UnderStat[] {
  const out: UnderStat[] = [];
  document.querySelectorAll('#underTotals .under-stat').forEach((el) => {
    const buckets: { label: string; value: string }[] = [];
    el.querySelectorAll('.under-bucket-row').forEach((row) => {
      buckets.push({
        label: row.querySelector('.under-bucket-label')?.textContent?.trim() || '',
        value: row.querySelector('.under-bucket-value')?.textContent?.trim().replace(/\s+/g, ' ') || '',
      });
    });
    out.push({
      label: el.querySelector('.under-stat-label')?.textContent?.trim() || '',
      value: el.querySelector('.under-stat-value')?.textContent?.trim() || '',
      meta: el.querySelector('.under-stat-meta')?.textContent?.trim().replace(/\s+/g, ' ') || '',
      buckets,
    });
  });
  return out;
}

function scrapeParkingStats(): { label: string; value: string }[] {
  const rows: { label: string; value: string }[] = [];
  const blurb = document.querySelector('#parkingSection .land-value-blurb, #parking-blurb, .parking-blurb');
  if (blurb?.textContent?.trim()) rows.push({ label: 'Overview', value: blurb.textContent.trim().replace(/\s+/g, ' ') });
  const grid = document.querySelectorAll('#parkingSection .totals-grid .label, #parkingSection .totals-grid .value');
  for (let i = 0; i + 1 < grid.length; i += 2) {
    const label = grid[i]?.textContent?.trim() || '';
    const value = grid[i + 1]?.textContent?.trim() || '';
    const row = (grid[i] as HTMLElement);
    if (label && value && value !== '–' && row.offsetParent !== null) rows.push({ label, value });
  }
  return rows;
}

function scrapeLandBlurb(): string {
  return document.getElementById('land-value-blurb')?.textContent?.trim().replace(/\s+/g, ' ') || '';
}

export async function generateCityReport(ctx: ReportContext, onStatus: (msg: string) => void): Promise<void> {
  const { jsPDF } = await import('jspdf');
  const originalTab = ctx.getTab();
  const tabs = ctx.tabsAvailable();

  // ── Collect snapshots + stats per view ──────────────────────────────────
  onStatus('Capturing 3D land value view…');
  ctx.setTab('main');
  await waitForIdle(ctx.map, 6000);
  const mainShot = await captureMap(ctx.map);
  const landBlurb = scrapeLandBlurb();

  let underShot: Awaited<ReturnType<typeof captureMap>> = null;
  let underStats: UnderStat[] = [];
  if (tabs.under) {
    onStatus('Capturing underused parcels view…');
    ctx.setTab('under');
    await waitForIdle(ctx.mapUnder, 8000);
    underShot = await captureMap(ctx.mapUnder);
    underStats = scrapeUnderStats();
  }

  let parkingShot: Awaited<ReturnType<typeof captureMap>> = null;
  let parkingStats: { label: string; value: string }[] = [];
  if (tabs.parking) {
    onStatus('Capturing surface parking view…');
    ctx.setTab('parking');
    // The parking workspace loads lazily; poll for its map, then let it settle.
    const deadline = Date.now() + 30000;
    let pmap = ctx.getParkingMap();
    while (!pmap && Date.now() < deadline) {
      await new Promise((r) => setTimeout(r, 500));
      pmap = ctx.getParkingMap();
    }
    if (pmap) {
      await waitForIdle(pmap, 12000);
      parkingShot = await captureMap(pmap);
      parkingStats = scrapeParkingStats();
    }
  }

  // Restore whatever the user was looking at.
  ctx.setTab(originalTab as any);

  onStatus('Composing PDF…');
  const logoWhite = await loadImage('/cle-logo-white.png');
  const logo2c = await loadImage('/cle-logo-2color.png');

  const doc = new jsPDF({ unit: 'mm', format: 'a4' });
  const city = ctx.cityLabel();
  const today = new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' });
  const shareUrl = window.location.href;
  let pageNum = 0;

  const addFooter = () => {
    doc.setFontSize(7.5);
    doc.setTextColor(...GRAY);
    doc.setFont('helvetica', 'normal');
    doc.text('civicmapper.org', MARGIN, PAGE_H - 8);
    doc.textWithLink('Center for Land Economics — landeconomics.org', PAGE_W / 2, PAGE_H - 8, {
      url: 'https://landeconomics.org', align: 'center',
    } as any);
    doc.textWithLink('progressandpoverty.substack.com', PAGE_W - MARGIN, PAGE_H - 8, {
      url: 'https://progressandpoverty.substack.com', align: 'right',
    } as any);
  };

  const addPageChrome = (title: string) => {
    pageNum += 1;
    if (pageNum > 1) doc.addPage();
    // Header band
    doc.setFillColor(...PURPLE);
    doc.rect(0, 0, PAGE_W, 24, 'F');
    if (logoWhite) {
      const h = 7;
      const w = h * (logoWhite.w / logoWhite.h);
      doc.addImage(logoWhite.data, 'PNG', MARGIN, 8.5, w, h);
    }
    doc.setTextColor(255, 255, 255);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11);
    doc.text(title, PAGE_W - MARGIN, 12, { align: 'right' });
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8.5);
    doc.text(`${city} · ${today}`, PAGE_W - MARGIN, 17.5, { align: 'right' });
    addFooter();
    return 32; // content start y
  };

  const addMapImage = (shot: { data: string; w: number; h: number } | null, y: number): number => {
    if (!shot) return y;
    const maxH = 110;
    let w = CONTENT_W;
    let h = w * (shot.h / shot.w);
    if (h > maxH) { h = maxH; w = h * (shot.w / shot.h); }
    const x = MARGIN + (CONTENT_W - w) / 2;
    doc.setDrawColor(210, 208, 220);
    doc.setLineWidth(0.3);
    doc.addImage(shot.data, 'JPEG', x, y, w, h);
    doc.rect(x, y, w, h, 'S');
    return y + h + 6;
  };

  const addSectionTitle = (text: string, y: number): number => {
    doc.setTextColor(...PURPLE);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(14);
    doc.text(text, MARGIN, y);
    return y + 7;
  };

  const addBody = (text: string, y: number, size = 10): number => {
    if (!text) return y;
    doc.setTextColor(...DARK);
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(size);
    const lines = doc.splitTextToSize(text, CONTENT_W);
    doc.text(lines, MARGIN, y);
    return y + lines.length * (size * 0.45) + 3;
  };

  // ── Page 1: cover / 3D land value ───────────────────────────────────────
  let y = addPageChrome('City Report');
  doc.setTextColor(...DARK);
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(26);
  doc.text(city, MARGIN, y + 6);
  y += 13;
  if (landBlurb) {
    doc.setFillColor(...LIGHT_BG);
    doc.roundedRect(MARGIN, y, CONTENT_W, 13, 2, 2, 'F');
    doc.setTextColor(...PURPLE);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(11.5);
    doc.text(doc.splitTextToSize(landBlurb, CONTENT_W - 10), MARGIN + 5, y + 8);
    y += 19;
  }
  y = addSectionTitle('Land Value in 3D', y + 2);
  y = addBody('Parcel heights and colors show assessed land value per square foot from local assessor records. Tall, dark parcels are the most valuable land in the city.', y);
  y = addMapImage(mainShot, y);

  // ── Page 2: Underused ───────────────────────────────────────────────────
  if (underShot || underStats.length) {
    y = addPageChrome('Underused Land');
    y = addSectionTitle('Vacant, Parking & Underdeveloped Parcels', y);
    y = addBody('Parcels whose improvements are small relative to their land value: vacant lots, whole-parcel surface parking, and underdeveloped sites (improvements worth less than half the parcel’s total value).', y);
    y = addMapImage(underShot, y);
    if (underStats.length) {
      for (const stat of underStats) {
        if (y > PAGE_H - 45) { y = addPageChrome('Underused Land (cont.)'); }
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(10.5);
        doc.setTextColor(...PURPLE);
        doc.text(stat.label, MARGIN, y);
        doc.setTextColor(...DARK);
        doc.text(stat.value, MARGIN + 60, y);
        y += 4.5;
        if (stat.meta) {
          doc.setFont('helvetica', 'normal');
          doc.setFontSize(8.5);
          doc.setTextColor(...GRAY);
          doc.text(doc.splitTextToSize(stat.meta, CONTENT_W - 4), MARGIN + 2, y);
          y += 4.5;
        }
        for (const b of stat.buckets) {
          doc.setFont('helvetica', 'normal');
          doc.setFontSize(8.5);
          doc.setTextColor(...GRAY);
          doc.text(`• ${b.label}`, MARGIN + 4, y);
          doc.setTextColor(...DARK);
          doc.text(b.value, MARGIN + 64, y);
          y += 4.2;
        }
        y += 2.5;
      }
    }
  }

  // ── Page 3: Parking ─────────────────────────────────────────────────────
  if (parkingShot || parkingStats.length) {
    y = addPageChrome('Surface Parking');
    y = addSectionTitle('Land Locked in Surface Parking', y);
    y = addBody('Surface parking lot footprints from OpenStreetMap joined with assessor land values — an estimate of how much valuable land sits under parked cars.', y);
    y = addMapImage(parkingShot, y);
    for (const row of parkingStats) {
      if (y > PAGE_H - 30) { y = addPageChrome('Surface Parking (cont.)'); }
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(9.5);
      doc.setTextColor(...GRAY);
      const labelLines = doc.splitTextToSize(row.label, 95);
      doc.text(labelLines, MARGIN, y);
      doc.setTextColor(...DARK);
      doc.setFont('helvetica', 'bold');
      doc.text(doc.splitTextToSize(row.value, 70), MARGIN + 100, y);
      y += Math.max(labelLines.length, 1) * 4.4 + 1.5;
    }
  }

  // ── Closing page: about + CTA ───────────────────────────────────────────
  y = addPageChrome('About');
  if (logo2c) {
    const h = 10;
    const w = h * (logo2c.w / logo2c.h);
    doc.addImage(logo2c.data, 'PNG', MARGIN, y, w, h);
    y += 16;
  }
  y = addSectionTitle('Land is a big deal.', y);
  y = addBody('This report was generated with Civic Mapper, a free tool from the Center for Land Economics. CLE conducts research and provides education to promote equitable assessments and foster sustainable development for the benefit of communities.', y);
  y += 2;
  y = addBody('Explore the interactive map:', y, 9.5);
  doc.setTextColor(...PURPLE);
  doc.setFontSize(9.5);
  doc.textWithLink(shareUrl, MARGIN, y, { url: shareUrl } as any);
  y += 9;
  y = addBody('To understand what these numbers mean for housing, taxes, and your city’s future, subscribe to Progress and Poverty, our newsletter on land value taxes, housing, and political economy:', y, 9.5);
  doc.setTextColor(...PURPLE);
  doc.textWithLink('progressandpoverty.substack.com', MARGIN, y, { url: 'https://progressandpoverty.substack.com' } as any);
  y += 9;
  y = addBody('Working on land value tax policy where you live? We’d love to hear from you:', y, 9.5);
  doc.setTextColor(...PURPLE);
  doc.textWithLink('greg@landeconomics.org', MARGIN, y, { url: 'mailto:greg@landeconomics.org' } as any);

  const slug = city.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') || 'city';
  doc.save(`civicmapper-${slug}-report.pdf`);
}
