import React, { useState, useEffect } from 'react';
import { useParams, useLocation } from 'react-router-dom';
import { useAuthorizedFetch } from './AuthContext';
import PageLayout from './PageLayout';

function MinutesPage() {
    const { fileId } = useParams();
    const location = useLocation();
    const authorizedFetch = useAuthorizedFetch();
    const [minutesContent, setMinutesContent] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [format, setFormat] = useState('text'); // Default format

    useEffect(() => {
        const queryParams = new URLSearchParams(location.search);
        const formatParam = queryParams.get('format') || 'text';
        setFormat(formatParam);

        const fetchMinutes = async () => {
            try {
                const response = await authorizedFetch(`/processed_files/${fileId}/minutes?format=${formatParam}`);
                if (!response.ok) {
                    throw new Error(`Błąd HTTP: ${response.status} ${response.statusText}`);
                }
                const content = await response.text(); // Minutes can be HTML or text
                setMinutesContent(content);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchMinutes();
    }, [fileId, location.search, authorizedFetch]);

    const pageTitle = `Protokoły spotkania (ID: ${fileId})`;
    const pageDescription = `Aktualny format: ${format.toUpperCase()}`;

    const formatActions =
        !loading && !error ? (
            <div className="flex items-center gap-2">
                <button
                    type="button"
                    onClick={() => setFormat('html')}
                    className={`rounded-md border px-4 py-2 text-sm font-semibold transition ${
                        format === 'html'
                            ? 'border-gray-900 bg-gray-900 text-white'
                            : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-100'
                    }`}
                >
                    Zobacz HTML
                </button>
                <button
                    type="button"
                    onClick={() => setFormat('text')}
                    className={`rounded-md border px-4 py-2 text-sm font-semibold transition ${
                        format === 'text'
                            ? 'border-gray-900 bg-gray-900 text-white'
                            : 'border-gray-300 bg-white text-gray-700 hover:bg-gray-100'
                    }`}
                >
                    Zobacz tekst
                </button>
            </div>
        ) : null;

    let content;
    if (loading) {
        content = (
            <div className="rounded-xl border border-gray-200 bg-white p-6 text-sm text-gray-600 shadow-sm">
                Ładowanie protokołów...
            </div>
        );
    } else if (error) {
        content = (
            <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-sm text-red-700 shadow-sm">
                Błąd: {error}
            </div>
        );
    } else if (!minutesContent) {
        content = (
            <div className="rounded-xl border border-gray-200 bg-white p-6 text-sm text-gray-600 shadow-sm">
                Brak protokołów.
            </div>
        );
    } else {
        content = (
            <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                {format === 'html' ? (
                    <div
                        className="minutes-html max-w-full overflow-x-auto"
                        dangerouslySetInnerHTML={{ __html: minutesContent }}
                    />
                ) : (
                    <pre className="whitespace-pre-wrap break-words overflow-x-auto max-w-full text-sm text-gray-800">
                        {minutesContent}
                    </pre>
                )}
            </div>
        );
    }

    return (
        <PageLayout title={pageTitle} description={pageDescription} actions={formatActions}>
            {content}
        </PageLayout>
    );
}

export default MinutesPage;
