const env = (import.meta as any).env ?? {};

function normalizeBase(base: string | undefined | null): string {
  const trimmed = typeof base === 'string' ? base.trim() : '';
  if (!trimmed) return '/api';
  return trimmed.replace(/\/$/, '') || '/api';
}

const API_BASE = normalizeBase(env.VITE_API_BASE);

function deriveEnvLabel(): string {
  const explicit = [env.VITE_ENV_LABEL, env.VITE_DEPLOY_ENV]
    .map((value) => (typeof value === 'string' ? value.trim() : ''))
    .find((value) => value);
  if (explicit) return explicit;
  if (env.PROD) return 'Prod';
  if (env.DEV) return 'Dev';
  if (typeof env.MODE === 'string' && env.MODE.trim()) return env.MODE.trim();
  return 'Local';
}

const ENV_LABEL = deriveEnvLabel();

export { API_BASE, ENV_LABEL };
