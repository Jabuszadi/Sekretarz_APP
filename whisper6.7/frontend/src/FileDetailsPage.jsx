import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useAuth } from './AuthContext'; // Importuj useAuth
import PageLayout from './PageLayout';

function FileDetailsPage() {
    const { fileId } = useParams();
    const [fileDetails, setFileDetails] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const { token } = useAuth(); // Pobierz token z kontekstu autoryzacji

    useEffect(() => {
        const fetchFileDetails = async () => {
            try {
                const response = await fetch(`/processed_files/${fileId}/details`, {
                    headers: {
                        'Authorization': `Bearer ${token}`
                    }
                });
                if (!response.ok) {
                    throw new Error(`Błąd HTTP: ${response.status} ${response.statusText}`);
                }
                const data = await response.json();
                setFileDetails(data);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchFileDetails();
    }, [fileId]);

    const pageTitle = 'Szczegóły pliku';
    const pageDescription = `Identyfikator: ${fileId}`;

    let content;
    if (loading) {
        content = (
            <div className="rounded-xl border border-gray-200 bg-white p-6 text-sm text-gray-600 shadow-sm">
                Ładowanie szczegółów pliku...
            </div>
        );
    } else if (error) {
        content = (
            <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-sm text-red-700 shadow-sm">
                Błąd: {error}
            </div>
        );
    } else if (!fileDetails) {
        content = (
            <div className="rounded-xl border border-gray-200 bg-white p-6 text-sm text-gray-600 shadow-sm">
                Brak szczegółów pliku.
            </div>
        );
    } else {
        content = (
            <div className="space-y-6">
                <section className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                    <h2 className="text-lg font-semibold text-gray-900">Metadane pliku</h2>
                    <dl className="mt-4 grid gap-4 sm:grid-cols-2">
                        <div className="space-y-1 text-sm text-gray-700">
                            <dt className="font-semibold text-gray-800">Nazwa pliku</dt>
                            <dd>{fileDetails.filename}</dd>
                        </div>
                        <div className="space-y-1 text-sm text-gray-700">
                            <dt className="font-semibold text-gray-800">ID pliku</dt>
                            <dd>{fileDetails.file_id}</dd>
                        </div>
                        <div className="space-y-1 text-sm text-gray-700">
                            <dt className="font-semibold text-gray-800">Hash</dt>
                            <dd className="break-all">{fileDetails.filehash}</dd>
                        </div>
                        <div className="space-y-1 text-sm text-gray-700">
                            <dt className="font-semibold text-gray-800">Status</dt>
                            <dd>{fileDetails.status || 'Status nieznany'}</dd>
                        </div>
                        <div className="space-y-1 text-sm text-gray-700 sm:col-span-2">
                            <dt className="font-semibold text-gray-800">Przetworzono</dt>
                            <dd>{new Date(fileDetails.processed_at).toLocaleString()}</dd>
                        </div>
                    </dl>
                </section>

                {fileDetails.meeting_data && (
                    <section className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                        <h2 className="text-lg font-semibold text-gray-900">Dane spotkania</h2>
                        <dl className="mt-4 grid gap-4 sm:grid-cols-2">
                            <div className="space-y-1 text-sm text-gray-700">
                                <dt className="font-semibold text-gray-800">ID transkrypcji</dt>
                                <dd>{fileDetails.meeting_data.transcript_id}</dd>
                            </div>
                            <div className="space-y-1 text-sm text-gray-700">
                                <dt className="font-semibold text-gray-800">ID protokołu</dt>
                                <dd>{fileDetails.meeting_data.minutes_id}</dd>
                            </div>
                            <div className="space-y-1 text-sm text-gray-700">
                                <dt className="font-semibold text-gray-800">Data spotkania</dt>
                                <dd>{fileDetails.meeting_data.meeting_date}</dd>
                            </div>
                            <div className="space-y-1 text-sm text-gray-700">
                                <dt className="font-semibold text-gray-800">Kolekcja Qdrant</dt>
                                <dd>{fileDetails.meeting_data.qdrant_collection_name}</dd>
                            </div>
                        </dl>
                    </section>
                )}

                {fileDetails.transcript && (
                    <section className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                        <h2 className="text-lg font-semibold text-gray-900">Transkrypcja</h2>
                        <dl className="mt-4 grid gap-4 sm:grid-cols-2">
                            <div className="space-y-1 text-sm text-gray-700">
                                <dt className="font-semibold text-gray-800">ID transkrypcji</dt>
                                <dd>{fileDetails.transcript.id}</dd>
                            </div>
                            <div className="space-y-1 text-sm text-gray-700 sm:col-span-2">
                                <dt className="font-semibold text-gray-800">Ścieżka pliku</dt>
                                <dd className="break-all">{fileDetails.transcript.content_path}</dd>
                            </div>
                        </dl>
                        <div className="mt-3">
                            <a
                                href={`/processed_files/${fileId}/transcript?token=${token}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-sm font-semibold text-gray-900 underline decoration-transparent hover:decoration-current"
                            >
                                Zobacz transkrypcję
                            </a>
                        </div>
                    </section>
                )}

                {fileDetails.minutes && (
                    <section className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                        <h2 className="text-lg font-semibold text-gray-900">Protokoły</h2>
                        <dl className="mt-4 grid gap-4 sm:grid-cols-2">
                            <div className="space-y-1 text-sm text-gray-700">
                                <dt className="font-semibold text-gray-800">ID protokołu</dt>
                                <dd>{fileDetails.minutes.id}</dd>
                            </div>
                            <div className="space-y-1 text-sm text-gray-700 sm:col-span-2">
                                <dt className="font-semibold text-gray-800">Ścieżka pliku</dt>
                                <dd className="break-all">{fileDetails.minutes.content_path}</dd>
                            </div>
                        </dl>
                        <div className="mt-3 space-x-3 text-sm font-semibold text-gray-900">
                            <a
                                href={`/processed_files/${fileId}/minutes?format=html&token=${token}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="underline decoration-transparent hover:decoration-current"
                            >
                                Zobacz protokoły (HTML)
                            </a>
                            <a
                                href={`/processed_files/${fileId}/minutes?format=text&token=${token}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="underline decoration-transparent hover:decoration-current"
                            >
                                Zobacz protokoły (tekst)
                            </a>
                        </div>
                    </section>
                )}
            </div>
        );
    }

    return (
        <PageLayout title={pageTitle} description={pageDescription}>
            {content}
        </PageLayout>
    );
}

export default FileDetailsPage;
