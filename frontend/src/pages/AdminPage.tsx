import { useState, useEffect, useCallback } from 'react';
import { useAppStore } from '../store/useAppStore';
import { listModels, updateAdminConfig, getAdminConfig } from '../api/client';
import type { ModelInfo, AdminConfig } from '../types';

/* ── Password Gate ── */
function PasswordGate({ onAuth }: { onAuth: (password: string) => void }) {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!password.trim()) {
      setError('נא להזין סיסמה');
      return;
    }
    setIsLoading(true);
    setError('');
    try {
      await getAdminConfig(password);
      onAuth(password);
    } catch (err: unknown) {
      const status = (err as { response?: { status?: number } })?.response?.status;
      if (status === 401) {
        setError('סיסמה שגויה — אין הרשאת גישה');
      } else {
        setError('שגיאה בחיבור לשרת');
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-1 items-center justify-center px-4 py-16">
      <div
        className="w-full max-w-sm rounded-xl border p-8"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          borderColor: 'var(--border-light)',
          boxShadow: 'var(--shadow-lg)',
        }}
      >
        <div className="mb-6 text-center">
          <div
            className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl"
            style={{ backgroundColor: 'var(--primary-50)' }}
          >
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" aria-hidden="true">
              <path
                d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z"
                stroke="var(--primary-600)"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </div>
          <h2
            className="text-xl font-bold"
            style={{
              fontFamily: 'var(--font-heading)',
              color: 'var(--text-primary)',
            }}
          >
            כניסת מנהל
          </h2>
          <p className="mt-1 text-sm" style={{ color: 'var(--text-muted)' }}>
            הזינו את סיסמת הניהול כדי לגשת להגדרות
          </p>
        </div>

        <form onSubmit={handleSubmit}>
          <label className="mb-1.5 block text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
            סיסמה
          </label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="mb-4 w-full rounded-lg border px-3 py-2.5 text-sm outline-none transition-colors"
            style={{
              borderColor: error ? 'var(--severity-high)' : 'var(--border-medium)',
              backgroundColor: 'var(--bg-primary)',
              color: 'var(--text-primary)',
            }}
            onFocus={(e) => {
              e.currentTarget.style.borderColor = 'var(--accent-500)';
              e.currentTarget.style.boxShadow = '0 0 0 3px color-mix(in srgb, var(--accent-500) 15%, transparent)';
            }}
            onBlur={(e) => {
              e.currentTarget.style.borderColor = error ? 'var(--severity-high)' : 'var(--border-medium)';
              e.currentTarget.style.boxShadow = 'none';
            }}
            placeholder="הזינו סיסמה..."
            dir="ltr"
            autoFocus
          />

          {error && (
            <p className="mb-3 text-sm font-medium" style={{ color: 'var(--severity-high)' }}>
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={isLoading}
            className="w-full cursor-pointer rounded-lg px-4 py-2.5 text-sm font-medium text-white transition-all disabled:opacity-60 disabled:cursor-not-allowed"
            style={{
              background: 'linear-gradient(135deg, var(--primary-700), var(--primary-800))',
              boxShadow: 'var(--shadow-sm)',
              border: 'none',
            }}
          >
            {isLoading ? (
              <span className="flex items-center justify-center gap-2">
                <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
                מתחבר...
              </span>
            ) : (
              'כניסה'
            )}
          </button>
        </form>
      </div>
    </div>
  );
}

/* ── Admin Config Panel ── */
function AdminConfigPanel({ password }: { password: string }) {
  const { adminConfig, fetchAdminConfig } = useAppStore();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [localConfig, setLocalConfig] = useState<AdminConfig | null>(null);
  const [isSaving, setIsSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);
  const [isLoadingModels, setIsLoadingModels] = useState(true);

  const loadData = useCallback(async () => {
    try {
      await fetchAdminConfig(password);
      const modelList = await listModels(password);
      setModels(modelList);
    } catch {
      // Error handled by store
    } finally {
      setIsLoadingModels(false);
    }
  }, [password, fetchAdminConfig]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    if (adminConfig) {
      setLocalConfig({ ...adminConfig });
    }
  }, [adminConfig]);

  const handleSave = async () => {
    if (!localConfig) return;
    setIsSaving(true);
    setSaveMessage(null);
    try {
      await updateAdminConfig(password, localConfig);
      await fetchAdminConfig(password);
      setSaveMessage({ type: 'success', text: 'ההגדרות נשמרו בהצלחה' });
      setTimeout(() => setSaveMessage(null), 3000);
    } catch (err: unknown) {
      const status = (err as { response?: { status?: number } })?.response?.status;
      if (status === 401) {
        setSaveMessage({ type: 'error', text: 'סיסמה שגויה — אין הרשאת גישה' });
      } else {
        setSaveMessage({ type: 'error', text: 'שגיאה בשמירת ההגדרות' });
      }
    } finally {
      setIsSaving(false);
    }
  };

  if (!localConfig) {
    return (
      <div className="flex items-center justify-center py-16">
        <span className="inline-block h-6 w-6 animate-spin rounded-full border-2 border-current/30 border-t-current" style={{ color: 'var(--accent-600)' }} />
        <span className="mr-3 text-sm" style={{ color: 'var(--text-muted)' }}>טוען הגדרות...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Model Selector */}
      <div
        className="rounded-xl border p-5"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          borderColor: 'var(--border-light)',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <label
          className="mb-1.5 block text-sm font-bold"
          style={{
            fontFamily: 'var(--font-heading)',
            color: 'var(--text-primary)',
          }}
        >
          מודל Bedrock
        </label>
        <p className="mb-3 text-xs" style={{ color: 'var(--text-muted)' }}>
          בחרו את מודל השפה לשכתוב טקסט
        </p>
        {isLoadingModels ? (
          <div className="flex items-center gap-2 text-sm" style={{ color: 'var(--text-muted)' }}>
            <span className="inline-block h-3.5 w-3.5 animate-spin rounded-full border-2 border-current/30 border-t-current" />
            טוען מודלים...
          </div>
        ) : (
          <select
            value={localConfig.bedrock_model_id}
            onChange={(e) =>
              setLocalConfig({ ...localConfig, bedrock_model_id: e.target.value })
            }
            className="w-full rounded-lg border px-3 py-2.5 text-sm outline-none transition-colors"
            style={{
              borderColor: 'var(--border-medium)',
              backgroundColor: 'var(--bg-primary)',
              color: 'var(--text-primary)',
            }}
            dir="ltr"
          >
            {models.length === 0 && (
              <option value={localConfig.bedrock_model_id}>
                {localConfig.bedrock_model_id}
              </option>
            )}
            {models.map((m) => (
              <option key={m.model_id} value={m.model_id}>
                {m.model_name} ({m.provider})
              </option>
            ))}
          </select>
        )}
      </div>

      {/* Threshold Slider */}
      <div
        className="rounded-xl border p-5"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          borderColor: 'var(--border-light)',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <label
          className="mb-1.5 block text-sm font-bold"
          style={{
            fontFamily: 'var(--font-heading)',
            color: 'var(--text-primary)',
          }}
        >
          סף חומרה
        </label>
        <p className="mb-3 text-xs" style={{ color: 'var(--text-muted)' }}>
          אבחנות עם חומרה מתחת לסף לא יוצגו למשתמש
        </p>
        <div className="flex items-center gap-4">
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={localConfig.severity_threshold}
            onChange={(e) =>
              setLocalConfig({
                ...localConfig,
                severity_threshold: parseFloat(e.target.value),
              })
            }
            className="flex-1"
            style={{ accentColor: 'var(--accent-600)' }}
          />
          <span
            className="w-12 rounded-lg px-2 py-1 text-center text-sm font-bold"
            style={{
              backgroundColor: 'var(--accent-50)',
              color: 'var(--accent-800)',
            }}
            dir="ltr"
          >
            {localConfig.severity_threshold.toFixed(2)}
          </span>
        </div>
      </div>

      {/* Display Settings */}
      <div
        className="rounded-xl border p-5"
        style={{
          backgroundColor: 'var(--bg-secondary)',
          borderColor: 'var(--border-light)',
          boxShadow: 'var(--shadow-sm)',
        }}
      >
        <h3
          className="mb-1.5 text-sm font-bold"
          style={{
            fontFamily: 'var(--font-heading)',
            color: 'var(--text-primary)',
          }}
        >
          הגדרות תצוגה
        </h3>
        <p className="mb-4 text-xs" style={{ color: 'var(--text-muted)' }}>
          מספר מרבי של פריטים להצגה בלוח הניתוח
        </p>

        <div className="space-y-4">
          <div>
            <label className="mb-1 block text-sm" style={{ color: 'var(--text-secondary)' }}>
              מספר אבחנות מרבי
            </label>
            <input
              type="number"
              min="1"
              max="10"
              value={localConfig.max_diagnoses_shown}
              onChange={(e) =>
                setLocalConfig({
                  ...localConfig,
                  max_diagnoses_shown: Math.max(1, Math.min(10, parseInt(e.target.value) || 1)),
                })
              }
              className="w-full rounded-lg border px-3 py-2 text-sm outline-none transition-colors"
              style={{
                borderColor: 'var(--border-medium)',
                backgroundColor: 'var(--bg-primary)',
                color: 'var(--text-primary)',
              }}
              dir="ltr"
            />
          </div>

          <div>
            <label className="mb-1 block text-sm" style={{ color: 'var(--text-secondary)' }}>
              מספר המלצות מרבי
            </label>
            <input
              type="number"
              min="1"
              max="10"
              value={localConfig.max_interventions_shown}
              onChange={(e) =>
                setLocalConfig({
                  ...localConfig,
                  max_interventions_shown: Math.max(1, Math.min(10, parseInt(e.target.value) || 1)),
                })
              }
              className="w-full rounded-lg border px-3 py-2 text-sm outline-none transition-colors"
              style={{
                borderColor: 'var(--border-medium)',
                backgroundColor: 'var(--bg-primary)',
                color: 'var(--text-primary)',
              }}
              dir="ltr"
            />
          </div>
        </div>
      </div>

      {/* Save Button + Message */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleSave}
          disabled={isSaving}
          className="cursor-pointer rounded-lg px-6 py-2.5 text-sm font-medium text-white transition-all disabled:opacity-60 disabled:cursor-not-allowed"
          style={{
            background: 'linear-gradient(135deg, var(--accent-600), var(--accent-700))',
            boxShadow: 'var(--shadow-sm)',
            border: 'none',
          }}
        >
          {isSaving ? (
            <span className="flex items-center gap-2">
              <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
              שומר...
            </span>
          ) : (
            'שמירת הגדרות'
          )}
        </button>

        {saveMessage && (
          <span
            className="text-sm font-medium"
            style={{
              color: saveMessage.type === 'success' ? 'var(--severity-low)' : 'var(--severity-high)',
            }}
          >
            {saveMessage.text}
          </span>
        )}
      </div>
    </div>
  );
}

/* ── Main Admin Page ── */
export default function AdminPage() {
  const [password, setPassword] = useState<string | null>(null);

  if (!password) {
    return <PasswordGate onAuth={setPassword} />;
  }

  return (
    <div className="mx-auto max-w-2xl flex-1 px-4 py-10 sm:px-6 lg:px-8">
      <div className="mb-8 text-center">
        <h1
          className="mb-3 text-3xl"
          style={{
            fontFamily: 'var(--font-heading)',
            color: 'var(--text-primary)',
          }}
        >
          ניהול מערכת
        </h1>
        <p className="text-base" style={{ color: 'var(--text-secondary)' }}>
          הגדרות מודל, סף חומרה ותצוגה
        </p>
      </div>

      <AdminConfigPanel password={password} />
    </div>
  );
}
