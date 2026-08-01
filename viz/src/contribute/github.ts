/**
 * GitHub URL builder for the contribute form.
 *
 * City proposals are raised as ISSUES on the main Civic Mapper repo — there is
 * no separate intake repo. The form generates a structured markdown proposal;
 * we pre-fill it into a new-issue URL when it fits, otherwise the user
 * downloads the file and pastes it into a blank issue.
 */

const MAIN_REPO = 'Center-for-Land-Economics/civicmapper-public';
const ISSUE_LABEL = 'city-proposal';

export interface GitHubSubmitOptions {
  stateCode: string;
  cityKey: string;
  markdown: string;
}

/**
 * Returns a GitHub URL that opens a new issue on the main repo with the
 * proposal pre-filled (works for shorter proposals — browser URL limits).
 *
 * For longer proposals, returns null and the caller should use the
 * download-then-paste approach.
 */
export function buildGitHubNewIssueUrl(opts: GitHubSubmitOptions): string | null {
  const title = `City proposal: ${opts.cityKey}-${opts.stateCode.toLowerCase()}`;
  const url =
    `https://github.com/${MAIN_REPO}/issues/new` +
    `?labels=${encodeURIComponent(ISSUE_LABEL)}` +
    `&title=${encodeURIComponent(title)}` +
    `&body=${encodeURIComponent(opts.markdown)}`;

  if (url.length > 8000) return null; // too long for the browser URL bar
  return url;
}

/**
 * Returns the URL to open a blank new city-proposal issue (used when the
 * generated markdown is too large to pre-fill; the user pastes it in).
 */
export function getNewIssueUrl(): string {
  return `https://github.com/${MAIN_REPO}/issues/new?labels=${encodeURIComponent(ISSUE_LABEL)}`;
}

/**
 * Triggers a browser download of the markdown file.
 */
export function downloadMarkdown(filename: string, content: string): void {
  const blob = new Blob([content], { type: 'text/markdown; charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

/**
 * Returns the URL for the main repo's contribution guide.
 */
export function getContributionGuideUrl(): string {
  return `https://github.com/${MAIN_REPO}/blob/main/CONTRIBUTING.md`;
}
