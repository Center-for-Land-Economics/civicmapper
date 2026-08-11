/**
 * Center for Land Economics footer + Substack subscribe modal.
 *
 * The footer is the site's quiet funnel: CLE logo → landeconomics.org,
 * "Land is a big deal" → in-page Substack subscribe (iframe modal, no new
 * tab), and a smaller advocacy reach-out mailto. Import and call
 * initCleFooter() on any page that should carry it; openSubscribeModal()
 * is exported separately so other CTAs (e.g. in the app sidebar) can open
 * the same modal.
 *
 * Subscribe-success detection mirrors the landeconomics.org implementation:
 * the sandboxed Substack iframe either fires a second `load` (internal
 * navigation to its confirmation page) or posts a message from
 * *.substack.com after the XHR submit — both are treated as success.
 */

const SUBSTACK_EMBED_URL = 'https://progressandpoverty.substack.com/embed';
const CONTACT_EMAIL = 'greg@landeconomics.org';

const STYLES = `
.cle-footer {
  background: #1b1440;
  color: #fff;
  padding: 56px 24px 28px;
  font-family: var(--font-sans, system-ui, sans-serif);
}
.cle-footer__inner {
  max-width: 1100px;
  margin: 0 auto;
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 40px;
}
.cle-footer__brand img { height: 40px; display: block; margin-bottom: 14px; }
.cle-footer__brand p {
  margin: 0;
  color: rgba(255,255,255,.72);
  font-size: .9rem;
  line-height: 1.6;
  max-width: 34ch;
}
.cle-footer__brand a.cle-footer__site {
  display: block;
  width: fit-content;
  margin-top: 12px;
  color: #b7e3e0;
  font-size: .9rem;
  text-decoration: none;
}
.cle-footer__brand a.cle-footer__site:hover { text-decoration: underline; }
.cle-footer__cta h3 {
  margin: 0 0 10px;
  font-size: 1.6rem;
  font-weight: 700;
  letter-spacing: -0.01em;
}
.cle-footer__cta p {
  margin: 0 0 18px;
  color: rgba(255,255,255,.72);
  font-size: .92rem;
  line-height: 1.6;
  max-width: 44ch;
}
.cle-footer__subscribe {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border: 0;
  border-radius: 9999px;
  background: #b7e3e0;
  color: #1b1440;
  font-family: inherit;
  font-weight: 700;
  font-size: .95rem;
  padding: 12px 28px;
  cursor: pointer;
  transition: background .15s ease;
}
.cle-footer__subscribe:hover { background: #d3efed; }
.cle-footer__advocacy {
  margin-top: 18px;
  font-size: .85rem;
  color: rgba(255,255,255,.6);
}
.cle-footer__advocacy a { color: rgba(255,255,255,.85); text-decoration: underline; text-underline-offset: 2px; }
.cle-footer__advocacy a:hover { color: #fff; }
.cle-footer__bottom {
  max-width: 1100px;
  margin: 44px auto 0;
  padding-top: 20px;
  border-top: 1px solid rgba(255,255,255,.12);
  display: flex;
  flex-wrap: wrap;
  gap: 10px 24px;
  justify-content: space-between;
  font-size: .78rem;
  color: rgba(255,255,255,.5);
}
.cle-footer__bottom a { color: rgba(255,255,255,.65); text-decoration: none; }
.cle-footer__bottom a:hover { color: #fff; text-decoration: underline; }

.cle-subscribe-overlay {
  position: fixed;
  inset: 0;
  z-index: 2000;
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 20px;
  background: rgba(10, 8, 28, .62);
  backdrop-filter: blur(4px);
}
.cle-subscribe-dialog {
  position: relative;
  width: 100%;
  max-width: 440px;
  background: #fff;
  color: #17181c;
  border-radius: 16px;
  padding: 28px;
  box-shadow: 0 25px 50px -12px rgba(0,0,0,.4);
  font-family: var(--font-sans, system-ui, sans-serif);
}
.cle-subscribe-dialog .cle-kicker {
  margin: 0 0 6px;
  font-size: .72rem;
  font-weight: 700;
  letter-spacing: .16em;
  text-transform: uppercase;
  color: #342877;
}
.cle-subscribe-dialog h2 { margin: 0 0 6px; font-size: 1.45rem; padding-right: 32px; }
.cle-subscribe-dialog .cle-sub { margin: 0 0 16px; font-size: .88rem; color: #5b5e6b; line-height: 1.5; }
.cle-subscribe-dialog iframe {
  display: block;
  width: 100%;
  height: 320px;
  border: 1px solid #e4e5ea;
  border-radius: 10px;
  background: #fff;
}
.cle-subscribe-close {
  position: absolute;
  right: 14px;
  top: 14px;
  width: 34px;
  height: 34px;
  border: 0;
  border-radius: 50%;
  background: transparent;
  color: #9195a1;
  font-size: 20px;
  line-height: 1;
  cursor: pointer;
}
.cle-subscribe-close:hover { background: #f0f1f4; color: #33353d; }
.cle-subscribe-thanks { display: flex; align-items: center; gap: 10px; font-size: 1.1rem; font-weight: 600; margin: 8px 0 4px; }
.cle-subscribe-thanks .tick {
  width: 34px; height: 34px; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  background: #d9f2e5; color: #0c7a4d; font-size: 18px;
}
`;

let stylesInjected = false;
function injectStyles() {
  if (stylesInjected) return;
  stylesInjected = true;
  const style = document.createElement('style');
  style.textContent = STYLES;
  document.head.appendChild(style);
}

export function openSubscribeModal(): void {
  injectStyles();

  const overlay = document.createElement('div');
  overlay.className = 'cle-subscribe-overlay';
  overlay.setAttribute('role', 'presentation');
  overlay.innerHTML = `
    <div class="cle-subscribe-dialog" role="dialog" aria-modal="true" aria-labelledby="cle-subscribe-heading">
      <button type="button" class="cle-subscribe-close" aria-label="Close">&#215;</button>
      <p class="cle-kicker">Progress and Poverty</p>
      <h2 id="cle-subscribe-heading">Subscribe to Progress and Poverty</h2>
      <p class="cle-sub">Writing on land value taxes, housing, and political economy from the Center for Land Economics.</p>
      <iframe
        src="${SUBSTACK_EMBED_URL}"
        title="Subscribe to Progress and Poverty"
        scrolling="no"
        sandbox="allow-scripts allow-forms allow-same-origin"
      ></iframe>
    </div>
  `;

  const dialog = overlay.querySelector('.cle-subscribe-dialog') as HTMLElement;
  const iframe = overlay.querySelector('iframe') as HTMLIFrameElement;
  const closeBtn = overlay.querySelector('.cle-subscribe-close') as HTMLButtonElement;

  const prevOverflow = document.body.style.overflow;
  document.body.style.overflow = 'hidden';

  let closed = false;
  const close = () => {
    if (closed) return;
    closed = true;
    document.body.style.overflow = prevOverflow;
    document.removeEventListener('keydown', onKeyDown);
    window.removeEventListener('message', onMessage);
    overlay.remove();
  };

  let submitted = false;
  const markSubmitted = () => {
    if (submitted) return;
    submitted = true;
    dialog.innerHTML = `
      <button type="button" class="cle-subscribe-close" aria-label="Close">&#215;</button>
      <p class="cle-kicker">Progress and Poverty</p>
      <div class="cle-subscribe-thanks"><span class="tick">&#10003;</span> Thanks for subscribing!</div>
      <p class="cle-sub">Check your inbox for a confirmation email from Substack.</p>
    `;
    (dialog.querySelector('.cle-subscribe-close') as HTMLButtonElement).addEventListener('click', close);
    window.setTimeout(close, 2500);
  };

  // Signal A: a second iframe load = Substack navigated to its confirmation page.
  let initialLoadSeen = false;
  iframe.addEventListener('load', () => {
    if (!initialLoadSeen) { initialLoadSeen = true; return; }
    markSubmitted();
  });

  // Signal B: Substack posts a message after a successful XHR submit.
  const onMessage = (event: MessageEvent) => {
    if (typeof event.origin !== 'string' || !event.origin.includes('substack.com')) return;
    if (event.source !== iframe.contentWindow) return;
    markSubmitted();
  };
  window.addEventListener('message', onMessage);

  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === 'Escape') { event.preventDefault(); close(); }
  };
  document.addEventListener('keydown', onKeyDown);

  overlay.addEventListener('click', (e) => { if (e.target === overlay) close(); });
  closeBtn.addEventListener('click', close);

  document.body.appendChild(overlay);
  closeBtn.focus();
}

export function initCleFooter(): void {
  injectStyles();

  const footer = document.createElement('footer');
  footer.className = 'cle-footer';
  footer.innerHTML = `
    <div class="cle-footer__inner">
      <div class="cle-footer__brand">
        <a href="https://landeconomics.org" target="_blank" rel="noopener" aria-label="Center for Land Economics">
          <img src="/cle-logo-white.svg" alt="Center for Land Economics">
        </a>
        <p>The Center for Land Economics conducts research and provides education to promote equitable assessments and foster sustainable development.</p>
        <a class="cle-footer__site" href="https://landeconomics.org" target="_blank" rel="noopener">landeconomics.org &rarr;</a>
        <a class="cle-footer__site" href="https://github.com/Center-for-Land-Economics/civicmapper" target="_blank" rel="noopener">Civic Mapper is open source &mdash; GitHub &rarr;</a>
      </div>
      <div class="cle-footer__cta">
        <h3>Land is a big deal.</h3>
        <p>To understand what these maps mean for housing, taxes, and your city&rsquo;s future, subscribe to Progress and Poverty &mdash; our writing on land value taxes, housing, and political economy.</p>
        <button type="button" class="cle-footer__subscribe" data-cle-subscribe>Subscribe</button>
        <p class="cle-footer__advocacy">Interested in land value tax advocacy where you live? <a href="mailto:${CONTACT_EMAIL}?subject=Land%20value%20tax%20advocacy">Reach out</a>.</p>
      </div>
    </div>
    <div class="cle-footer__bottom">
      <span>&copy; ${new Date().getFullYear()} Center for Land Economics &middot; Civic Mapper is a CLE project</span>
      <span>
        <a href="https://progressandpoverty.substack.com" target="_blank" rel="noopener">Substack</a>
        &nbsp;&middot;&nbsp;
        <a href="https://putitonamap.com" target="_blank" rel="noopener">Put It On A Map</a>
        &nbsp;&middot;&nbsp;
        <a href="https://givebutter.com/wK8u7p" target="_blank" rel="noopener">Donate</a>
        &nbsp;&middot;&nbsp;
        <a href="mailto:${CONTACT_EMAIL}">Contact</a>
      </span>
    </div>
  `;

  footer.querySelector('[data-cle-subscribe]')?.addEventListener('click', openSubscribeModal);
  document.body.appendChild(footer);
}
