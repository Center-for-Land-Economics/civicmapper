/**
 * Google Analytics initialisation.
 * VITE_GA_ID is injected at build time from Azure Key Vault (prod only).
 * In dev / staging the variable is absent, so this module is a no-op.
 */
const gaId = (import.meta as any).env?.VITE_GA_ID as string | undefined;

if (gaId) {
  // Load the gtag.js script
  const script = document.createElement('script');
  script.async = true;
  script.src = `https://www.googletagmanager.com/gtag/js?id=${gaId}`;
  document.head.appendChild(script);

  // Initialise the data layer exactly as Google's official snippet requires.
  // gtag.js reads dataLayer entries as Arguments objects — spread params break this.
  (window as any).dataLayer = (window as any).dataLayer || [];
  (window as any).gtag = function gtag() { (window as any).dataLayer.push(arguments); };
  (window as any).gtag('js', new Date());
  (window as any).gtag('config', gaId);
}
