import React, { useEffect, useMemo, useState } from 'react';
import {
  encryptAndStoreToken,
  decryptStoredToken,
  deleteStoredToken,
  listStoredProviders,
} from './utils/encryptedStorage';
import { useApiTokens } from './ApiTokensContext';
import PageLayout from './PageLayout';

const PROVIDER_CONFIG = {
  gemini: {
    label: 'Google Gemini',
    description: 'Używane do transkrypcji i generowania podsumowań.',
  },
  assemblyai: {
    label: 'AssemblyAI',
    description: 'Wspiera alternatywnych dostawców transkrypcji audio.',
  },
  openai: {
    label: 'OpenAI',
    description: 'Wykorzystywane do modeli GPT oraz Whisper API.',
  },
};

const MASK_PLACEHOLDER = '••••••••';

function ApiKeysPage() {
  const [passphrase, setPassphrase] = useState('');
  const [draftTokens, setDraftTokens] = useState({
    gemini: '',
    assemblyai: '',
    openai: '',
  });
  const [storedProviders, setStoredProviders] = useState({});
  const [revealedProviders, setRevealedProviders] = useState({});
  const [status, setStatus] = useState(null);
  const { tokens: activeTokens, setToken: setActiveToken } = useApiTokens();

  const providerEntries = useMemo(() => Object.entries(PROVIDER_CONFIG), []);

  useEffect(() => {
    setStoredProviders(listStoredProviders());
  }, []);

  const resetStatus = () => setStatus(null);

  const handleDraftChange = (provider, value) => {
    resetStatus();
    setDraftTokens((prev) => ({
      ...prev,
      [provider]: value,
    }));
  };

  const refreshStoredProviders = () => {
    setStoredProviders(listStoredProviders());
  };

  const handleSave = async (provider) => {
    resetStatus();
    const trimmedValue = (draftTokens[provider] || '').trim();
    const isEditingExisting = storedProviders[provider] && revealedProviders[provider];
    const isNewEntry = !storedProviders[provider];
    if ((!trimmedValue || trimmedValue === MASK_PLACEHOLDER) && (isEditingExisting || isNewEntry)) {
      setStatus({ type: 'info', message: 'Najpierw wpisz lub odsłoń klucz przed zapisaniem.' });
      return;
    }

    try {
      await encryptAndStoreToken(provider, trimmedValue, passphrase);
      setStatus({ type: 'success', message: `Klucz ${PROVIDER_CONFIG[provider].label} zapisany lokalnie.` });
      setDraftTokens((prev) => ({ ...prev, [provider]: '' }));
      setRevealedProviders((prev) => ({ ...prev, [provider]: false }));
      setActiveToken(provider, trimmedValue || undefined);
      refreshStoredProviders();
    } catch (error) {
      setStatus({ type: 'error', message: error.message || 'Nie udało się zapisać klucza.' });
    }
  };

  const handleReveal = async (provider) => {
    resetStatus();
    try {
      const token = (await decryptStoredToken(provider, passphrase)) || '';
      setDraftTokens((prev) => ({ ...prev, [provider]: token }));
      setRevealedProviders((prev) => ({ ...prev, [provider]: true }));
      setActiveToken(provider, token.trim() || undefined);
      setStatus({ type: 'info', message: `Klucz ${PROVIDER_CONFIG[provider].label} został odszyfrowany.` });
    } catch (error) {
      setStatus({ type: 'error', message: error.message || 'Nie udało się odszyfrować klucza.' });
    }
  };

  const handleDelete = (provider) => {
    resetStatus();
    deleteStoredToken(provider);
    setDraftTokens((prev) => ({ ...prev, [provider]: '' }));
    setRevealedProviders((prev) => ({ ...prev, [provider]: false }));
    setActiveToken(provider, undefined);
    refreshStoredProviders();
    setStatus({ type: 'success', message: `Klucz ${PROVIDER_CONFIG[provider].label} został usunięty.` });
  };

  return (
    <PageLayout
      title="Klucze API (lokalne)"
      description="Klucze są przechowywane zaszyfrowane w Twojej przeglądarce. My ich nie zapisujemy."
    >
      <div className="mx-auto w-full max-w-3xl space-y-6">

        <section className="rounded-md border border-gray-300 bg-gray-100 px-4 py-3 text-sm text-gray-900">
          <ul className="list-disc list-inside space-y-1">
            <li>Klucze są zaszyfrowane w <code>localStorage</code>, odszyfrujesz je tylko tym hasłem.</li>
            <li>
              Hasło nie jest nigdzie wysyłane – jeśli je zgubisz, będziesz musiał wprowadzić klucze ponownie.
            </li>
            <li>Po wylogowaniu lub zmianie przeglądarki klucze znikają – to celowy mechanizm bezpieczeństwa.</li>
          </ul>
        </section>

        <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
          <label className="block text-sm font-medium text-gray-700" htmlFor="passphrase-input">
            Hasło szyfrujące
          </label>
          <input
            id="passphrase-input"
            type="password"
            className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-gray-900 focus:outline-none focus:ring-1 focus:ring-gray-900"
            placeholder="Wpisz hasło (min. 6 znaków)"
            value={passphrase}
            onChange={(event) => {
              resetStatus();
              setPassphrase(event.target.value);
            }}
            autoComplete="off"
          />
        </div>

        {status && (
          <div
            className={`rounded-md px-4 py-3 text-sm ${
              status.type === 'error'
                ? 'border border-red-200 bg-red-50 text-red-800'
                : status.type === 'success'
                  ? 'border border-green-200 bg-green-50 text-green-800'
                  : 'border border-gray-300 bg-gray-100 text-gray-900'
            }`}
          >
            {status.message}
          </div>
        )}

        <div className="space-y-6">
          {providerEntries.map(([providerKey, providerData]) => {
            const stored = storedProviders[providerKey];
            const hasStored = Boolean(stored);
            const isActive = Boolean(activeTokens[providerKey]);
            const isRevealed = Boolean(revealedProviders[providerKey]);
            const currentDraftValue = draftTokens[providerKey] || '';
            const trimmedDraftValue = currentDraftValue.trim();
            const displayValue = isRevealed
              ? currentDraftValue
              : hasStored
                ? MASK_PLACEHOLDER
                : currentDraftValue;
            const canSave =
              passphrase.length >= 6
              && (
                (!hasStored && trimmedDraftValue.length > 0)
                || (hasStored && isRevealed && trimmedDraftValue.length > 0)
              );

            return (
              <section
                key={providerKey}
                className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm"
              >
                <header className="flex items-start justify-between">
                  <div>
                    <h2 className="text-lg font-semibold text-gray-900">{providerData.label}</h2>
                    <p className="text-sm text-gray-600">{providerData.description}</p>
                  </div>
                  {hasStored ? (
                    <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-medium text-emerald-700">
                      Zapisano lokalnie
                    </span>
                  ) : (
                    <span className="rounded-full bg-gray-100 px-3 py-1 text-xs font-medium text-gray-600">
                      Brak klucza
                    </span>
                  )}
                  {isActive && (
                    <span className="ml-2 rounded-full bg-gray-900 px-3 py-1 text-xs font-medium text-white">
                      Aktywny w sesji
                    </span>
                  )}
                </header>

                {hasStored && stored.updatedAt && (
                  <p className="mt-2 text-xs text-gray-500">
                    Ostatnia aktualizacja: {new Date(stored.updatedAt).toLocaleString()}
                  </p>
                )}

                <div className="mt-4 space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700" htmlFor={`${providerKey}-token`}>
                      Klucz API
                    </label>
                    <input
                      id={`${providerKey}-token`}
                      type={revealedProviders[providerKey] ? 'text' : 'password'}
                      autoComplete="off"
                      className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-gray-900 focus:outline-none focus:ring-1 focus:ring-gray-900"
                      placeholder="Wklej lub odszyfruj klucz"
                      value={displayValue}
                      readOnly={hasStored && !isRevealed}
                      onChange={(event) => handleDraftChange(providerKey, event.target.value)}
                    />
                  </div>

                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      type="button"
                      onClick={() => handleSave(providerKey)}
                      className="inline-flex items-center rounded-md bg-gray-900 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-gray-800 disabled:cursor-not-allowed disabled:bg-gray-500"
                      disabled={!canSave}
                    >
                      Zapisz lokalnie
                    </button>

                    <button
                      type="button"
                      onClick={() => handleReveal(providerKey)}
                      className="inline-flex items-center rounded-md border border-gray-400 bg-gray-100 px-4 py-2 text-sm font-semibold text-gray-900 transition hover:bg-gray-200 disabled:cursor-not-allowed disabled:text-gray-400"
                      disabled={!hasStored || passphrase.length < 6}
                    >
                      Odszyfruj zapisany
                    </button>

                    <button
                      type="button"
                      onClick={() => handleDelete(providerKey)}
                      className="inline-flex items-center rounded-md border border-red-200 bg-red-50 px-4 py-2 text-sm font-semibold text-red-700 transition hover:bg-red-100 disabled:cursor-not-allowed disabled:text-red-300"
                      disabled={!hasStored}
                    >
                      Usuń klucz
                    </button>
                  </div>
                </div>
              </section>
            );
          })}
        </div>
      </div>
    </PageLayout>
  );
}

export default ApiKeysPage;

