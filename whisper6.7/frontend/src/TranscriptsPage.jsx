import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useAuthorizedFetch, useAuth } from './AuthContext';
import { useApiTokens } from './ApiTokensContext';
import PageLayout from './PageLayout';

const formatDateTime = (value) => {
    if (!value) {
        return 'Brak danych';
    }
    try {
        const date = typeof value === 'string' ? new Date(value) : value;
        if (Number.isNaN(date.getTime())) {
            return value;
        }
        return date.toLocaleString();
    } catch {
        return value;
    }
};

function TranscriptsPage() {
    const authorizedFetch = useAuthorizedFetch();
    const { token } = useAuth();
    const { tokens: sessionTokens } = useApiTokens();
    const [transcripts, setTranscripts] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [actionMessage, setActionMessage] = useState(null);
    const [actionError, setActionError] = useState(null);
    const [regenerateState, setRegenerateState] = useState({
        open: false,
        fileId: null,
        filename: '',
        customPrompt: '',
        outputName: '',
        submitting: false,
    });

    const fetchTranscripts = useCallback(async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await authorizedFetch('/user/transcripts/');
            if (!response.ok) {
                throw new Error(`Błąd ${response.status}: ${response.statusText}`);
            }
            const rawBody = await response.text();
            let parsed = [];
            if (rawBody) {
                try {
                    const parsedJson = JSON.parse(rawBody);
                    if (Array.isArray(parsedJson)) {
                        parsed = parsedJson;
                    }
                } catch (jsonError) {
                    console.warn('Nie udało się zdekodować listy transkryptów, przyjmuję pustą listę:', jsonError);
                }
            }
            setTranscripts(parsed);
        } catch (err) {
            setError(err.message || 'Nie udało się pobrać listy transkryptów.');
        } finally {
            setLoading(false);
        }
    }, [authorizedFetch]);

    useEffect(() => {
        fetchTranscripts();
    }, [fetchTranscripts]);

    const handleOpenUrlWithToken = useCallback(
        (urlPath, params = {}) => {
            if (!token) {
                setActionError('Brak tokena uwierzytelniającego. Zaloguj się ponownie.');
                return;
            }
            const url = new URL(urlPath, window.location.origin);
            Object.entries(params).forEach(([key, value]) => {
                if (value !== undefined && value !== null) {
                    url.searchParams.set(key, String(value));
                }
            });
            url.searchParams.set('token', token);
            window.open(url.toString(), '_blank', 'noopener');
        },
        [token],
    );

    const handleDelete = useCallback(
        async (fileId, filename) => {
            const confirmed = window.confirm(`Czy na pewno chcesz usunąć transkrypt "${filename}"?`);
            if (!confirmed) {
                return;
            }
            setActionError(null);
            setActionMessage(null);
            try {
                const response = await authorizedFetch(`/user/transcripts/${fileId}`, { method: 'DELETE' });
                if (!response.ok) {
                    const details = await response.json().catch(() => ({}));
                    throw new Error(details.detail || `Błąd ${response.status}`);
                }
                setActionMessage('Transkrypt został usunięty.');
                await fetchTranscripts();
            } catch (err) {
                setActionError(err.message || 'Nie udało się usunąć transkryptu.');
            }
        },
        [authorizedFetch, fetchTranscripts],
    );

    const openRegenerateModal = useCallback((fileId, filename) => {
        setRegenerateState({
            open: true,
            fileId,
            filename,
            customPrompt: '',
            outputName: '',
            submitting: false,
        });
        setActionError(null);
        setActionMessage(null);
    }, []);

    const closeRegenerateModal = useCallback(() => {
        setRegenerateState((prev) => ({ ...prev, open: false, submitting: false }));
    }, []);

    const handleRegenerate = useCallback(async () => {
        if (!regenerateState.fileId) {
            return;
        }
        setRegenerateState((prev) => ({ ...prev, submitting: true }));
        setActionError(null);
        setActionMessage(null);
        try {
            const providerTokensPayload = {};
            ['gemini', 'assemblyai', 'openai'].forEach((key) => {
                const rawToken = sessionTokens?.[key];
                if (typeof rawToken === 'string') {
                    const trimmed = rawToken.trim();
                    if (trimmed) {
                        providerTokensPayload[key] = trimmed;
                    }
                }
            });

            const requestBody = {
                custom_prompt: regenerateState.customPrompt || null,
                output_name: regenerateState.outputName || null,
            };
            if (Object.keys(providerTokensPayload).length > 0) {
                requestBody.provider_tokens = providerTokensPayload;
            }

            const response = await authorizedFetch(
                `/user/transcripts/${regenerateState.fileId}/regenerate_minutes`,
                {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(requestBody),
                },
            );
            if (!response.ok) {
                const details = await response.json().catch(() => ({}));
                throw new Error(details.detail || `Błąd ${response.status}`);
            }
            setActionMessage('Podsumowanie zostało wygenerowane ponownie.');
            closeRegenerateModal();
            await fetchTranscripts();
        } catch (err) {
            setActionError(err.message || 'Nie udało się wygenerować podsumowania.');
            setRegenerateState((prev) => ({ ...prev, submitting: false }));
        }
    }, [authorizedFetch, regenerateState, fetchTranscripts, closeRegenerateModal, sessionTokens]);

    const transcriptsToDisplay = useMemo(() => transcripts, [transcripts]);

    return (
        <PageLayout
            title="Transkrypcje"
            description="Lista wszystkich przetworzonych plików. Możesz ponownie wygenerować podsumowanie lub usunąć wpis."
            actions={
                <button
                    onClick={fetchTranscripts}
                    className="inline-flex items-center rounded-md bg-gray-900 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-gray-800 disabled:opacity-50"
                    disabled={loading}
                >
                    Odśwież listę
                </button>
            }
        >

            {actionMessage && (
                <div className="rounded-md border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700">
                    {actionMessage}
                </div>
            )}
            {actionError && (
                <div className="rounded-md border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                    {actionError}
                </div>
            )}

            {loading ? (
                <div className="text-gray-600">Ładowanie transkryptów...</div>
            ) : error ? (
                <div className="text-red-600">Błąd: {error}</div>
            ) : transcriptsToDisplay.length === 0 ? (
                <div className="rounded-md border border-gray-200 bg-white p-6 text-center text-gray-600 shadow-sm">
                    Nie znaleziono żadnych transkryptów. Prześlij plik, aby rozpocząć.
                </div>
            ) : (
                <div className="overflow-x-auto rounded-xl border border-gray-200 bg-white shadow-sm">
                    <table className="min-w-full divide-y divide-gray-200">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-500">Nazwa pliku</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-500">Wersja</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-500">Status</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-500">Zaktualizowano</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-500">Streszczenie</th>
                                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-gray-500">Akcje</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200">
                            {transcriptsToDisplay.map((item) => {
                                const rowKey = item.minutes_id
                                    ? `${item.file_id}-minutes-${item.minutes_id}`
                                    : `file-${item.file_id}`;
                                const isLatest = Boolean(item.is_latest_minutes);
                                return (
                                    <tr key={rowKey} className={isLatest ? '' : 'bg-slate-50'}>
                                        <td className="px-4 py-3 text-sm font-medium text-gray-900">{item.filename || `Plik #${item.file_id}`}</td>
                                        <td className="px-4 py-3 text-sm text-gray-700">
                                            {item.has_minutes ? (
                                                <span
                                                    className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                                                        isLatest
                                                            ? 'bg-gray-200 text-gray-900'
                                                            : 'bg-gray-200 text-gray-700'
                                                    }`}
                                                >
                                                    Wersja {item.minutes_version}
                                                    {isLatest ? ' (najnowsza)' : ''}
                                                </span>
                                            ) : (
                                                <span className="inline-flex items-center rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-semibold text-gray-600">
                                                    Brak
                                                </span>
                                            )}
                                        </td>
                                    <td className="px-4 py-3 text-sm text-gray-700">
                                        <span
                                            className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                                                item.status === 'completed'
                                                    ? 'bg-green-100 text-green-800'
                                                    : item.status === 'error'
                                                    ? 'bg-red-100 text-red-800'
                                                    : 'bg-gray-100 text-gray-800'
                                            }`}
                                        >
                                            {item.status || 'nieznany'}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 text-sm text-gray-700">{formatDateTime(item.processed_at)}</td>
                                        <td className="px-4 py-3 text-sm text-gray-700">
                                            {item.has_minutes ? (
                                                <div className="space-y-1">
                                                    <span className="inline-flex items-center rounded-full bg-gray-200 px-2.5 py-0.5 text-xs font-semibold text-gray-900">
                                                        Gotowe {formatDateTime(item.minutes_created_at || item.minutes_generated_at)}
                                                    </span>
                                                    {item.minutes_summary_preview && (
                                                        <p
                                                            className="text-xs text-gray-600 whitespace-pre-line overflow-hidden"
                                                            style={{
                                                                display: '-webkit-box',
                                                                WebkitLineClamp: 3,
                                                                WebkitBoxOrient: 'vertical',
                                                            }}
                                                        >
                                                            {item.minutes_summary_preview}
                                                            {item.minutes_summary_length >
                                                            (item.minutes_summary_preview?.length || 0)
                                                                ? '…'
                                                                : ''}
                                                        </p>
                                                    )}
                                                </div>
                                            ) : (
                                                <span className="inline-flex items-center rounded-full bg-gray-100 px-2.5 py-0.5 text-xs font-semibold text-gray-600">
                                                    Brak
                                                </span>
                                            )}
                                        </td>
                                    <td className="px-4 py-3 text-sm text-gray-700">
                                        <div className="flex flex-wrap items-center gap-2">
                                            <button
                                                type="button"
                                                className="rounded-md border border-gray-700 bg-white px-3 py-1 text-xs font-semibold text-gray-900 shadow-sm transition hover:bg-gray-100 hover:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-400"
                                                onClick={() => handleOpenUrlWithToken(`/processed_files/${item.file_id}/transcript`)}
                                            >
                                                Zobacz transkrypcję
                                            </button>
                                            <button
                                                type="button"
                                                className="rounded-md border border-gray-700 bg-white px-3 py-1 text-xs font-semibold text-gray-900 shadow-sm transition hover:bg-gray-100 hover:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-400 disabled:cursor-not-allowed disabled:opacity-50"
                                                onClick={() =>
                                                    handleOpenUrlWithToken(`/processed_files/${item.file_id}/minutes`, {
                                                        format: 'html',
                                                        minutes_id: item.minutes_id ?? undefined,
                                                    })
                                                }
                                                disabled={!item.has_minutes}
                                            >
                                                Zobacz podsumowanie
                                            </button>
                                            <button
                                                type="button"
                                                className="rounded-md bg-gray-900 px-3 py-1 text-xs font-semibold text-white shadow-sm transition hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-400 disabled:cursor-not-allowed disabled:opacity-50"
                                                onClick={() =>
                                                    openRegenerateModal(item.file_id, item.filename || `Plik #${item.file_id}`)
                                                }
                                                disabled={item.has_minutes && !isLatest}
                                            >
                                                Ponownie generuj
                                            </button>
                                            <button
                                                type="button"
                                                className="rounded-md border border-red-600 bg-white px-3 py-1 text-xs font-semibold text-red-700 shadow-sm transition hover:bg-red-50 hover:border-red-700 focus:outline-none focus:ring-2 focus:ring-red-400"
                                                onClick={() => handleDelete(item.file_id, item.filename || `Plik #${item.file_id}`)}
                                            >
                                                Usuń
                                            </button>
                                        </div>
                                    </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            )}

            {regenerateState.open && (
                <div className="fixed inset-0 z-40 flex items-center justify-center bg-black/40 px-4">
                    <div className="w-full max-w-lg rounded-lg bg-white shadow-xl">
                        <div className="border-b border-gray-200 px-6 py-4">
                            <h2 className="text-lg font-semibold text-gray-900">
                                Ponowne generowanie podsumowania — {regenerateState.filename}
                            </h2>
                            <p className="text-sm text-gray-600">
                                Wprowadź własny prompt (opcjonalnie), aby dostosować treść streszczenia.
                            </p>
                        </div>
                        <div className="space-y-4 px-6 py-4">
                            <div>
                                <label className="block text-xs font-semibold uppercase tracking-wide text-gray-500">
                                    Custom prompt
                                </label>
                                <textarea
                                    rows={5}
                                    value={regenerateState.customPrompt}
                                    onChange={(event) =>
                                        setRegenerateState((prev) => ({ ...prev, customPrompt: event.target.value }))
                                    }
                                    className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-900/30"
                                    placeholder="Opcjonalnie: własny prompt dla LLM"
                                />
                            </div>
                            <div>
                                <label className="block text-xs font-semibold uppercase tracking-wide text-gray-500">
                                    Nazwa wyjściowa (opcjonalnie)
                                </label>
                                <input
                                    type="text"
                                    value={regenerateState.outputName}
                                    onChange={(event) =>
                                        setRegenerateState((prev) => ({ ...prev, outputName: event.target.value }))
                                    }
                                    className="mt-1 w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-900/30"
                                    placeholder="Niestandardowa nazwa pliku wyjściowego"
                                />
                            </div>
                        </div>
                        <div className="flex items-center justify-end gap-3 border-t border-gray-200 px-6 py-4">
                            <button
                                type="button"
                                className="rounded-md border border-gray-200 px-4 py-2 text-sm font-semibold text-gray-700 transition hover:bg-gray-100"
                                onClick={closeRegenerateModal}
                                disabled={regenerateState.submitting}
                            >
                                Anuluj
                            </button>
                            <button
                                type="button"
                                className="inline-flex items-center rounded-md bg-gray-900 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-gray-800 disabled:opacity-50"
                                onClick={handleRegenerate}
                                disabled={regenerateState.submitting}
                            >
                                {regenerateState.submitting ? 'Generowanie...' : 'Generuj ponownie'}
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </PageLayout>
    );
}

export default TranscriptsPage;

