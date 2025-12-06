import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useAuth, useAuthorizedFetch } from './AuthContext';
import PageLayout from './PageLayout';

function ProcessedBatchDetailsPage() {
    const { batchJobId } = useParams();
    const { isAuthenticated, token } = useAuth();
    const authorizedFetch = useAuthorizedFetch();
    const [batchDetails, setBatchDetails] = useState(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        console.log('ProcessedBatchDetailsPage useEffect triggered.');
        console.log('isAuthenticated in ProcessedBatchDetailsPage useEffect:', isAuthenticated);
        console.log('Current token in ProcessedBatchDetailsPage useEffect:', token);

        if (!isAuthenticated) {
            setError('Musisz być zalogowany, aby przeglądać szczegóły partii.');
            setLoading(false);
            return;
        }

        const fetchBatchDetails = async () => {
            const requestUrl = `/processed_batches/${batchJobId}/details`;
            console.log('Fetching batch details from URL:', requestUrl);
            try {
                const response = await authorizedFetch(requestUrl);
                console.log('API response status:', response.status);
                if (response.ok) {
                    const data = await response.json();
                    console.log('API response data:', data);
                    setBatchDetails(data);
                } else {
                    const errorData = await response.json();
                    console.error('API error data:', errorData);
                    setError(errorData.detail || 'Nie udało się pobrać szczegółów partii.');
                }
            } catch (err) {
                console.error('Network or fetch error:', err);
                setError('Wystąpił błąd sieci: ' + err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchBatchDetails();
    }, [batchJobId, isAuthenticated, authorizedFetch, token]); // Dodaj token do zależności

    const pageTitle = 'Szczegóły przetwarzania partii';
    const pageDescription = batchDetails
        ? `Identyfikator: ${batchDetails.batch_job_id}`
        : `Identyfikator: ${batchJobId}`;

    let content;
    if (loading) {
        content = (
            <div className="rounded-xl border border-gray-200 bg-white p-6 text-sm text-gray-600 shadow-sm">
                Ładowanie szczegółów partii...
            </div>
        );
    } else if (error) {
        content = (
            <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-sm text-red-700 shadow-sm">
                Błąd: {error}
            </div>
        );
    } else if (!batchDetails) {
        content = (
            <div className="rounded-xl border border-gray-200 bg-white p-6 text-sm text-gray-600 shadow-sm">
                Brak danych dla tej partii.
            </div>
        );
    } else {
        content = (
            <>
                <section className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                    <div className="space-y-2">
                        <p className="text-sm text-gray-700">
                            <strong className="text-gray-800">Ogólny status:</strong>{' '}
                            <span className="font-semibold text-green-600">
                                {batchDetails.batch_status === 'completed' ? 'Zakończono' : batchDetails.batch_status}
                            </span>
                        </p>
                        <p className="text-sm text-gray-700">
                            <strong className="text-gray-800">Całkowita liczba plików:</strong> {batchDetails.total_files}
                        </p>
                    </div>
                </section>

                <section className="space-y-4 rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                    <h2 className="text-lg font-semibold text-gray-900">Przetworzone pliki</h2>
                    <div className="space-y-4">
                        {batchDetails.files.map((file, index) => (
                            <div key={index} className="space-y-2 rounded-lg border border-gray-200 bg-gray-50 p-4 shadow-sm">
                                <h3 className="text-base font-semibold text-gray-900">
                                    Plik: {file.filename || 'Nazwa pliku nieznana'}
                                </h3>
                                <p className="text-sm text-gray-700">
                                    <strong>Status:</strong>{' '}
                                    <span
                                        className={`font-semibold ${
                                            file.status === 'completed'
                                                ? 'text-green-600'
                                                : file.status === 'error'
                                                ? 'text-red-600'
                                                : 'text-yellow-600'
                                        }`}
                                    >
                                        {file.status === 'completed'
                                            ? 'Zakończono pomyślnie'
                                            : file.status === 'processing'
                                            ? 'W trakcie przetwarzania'
                                            : file.status === 'uploaded'
                                            ? 'Wgrano (oczekuje na przetwarzanie)'
                                            : file.status === 'error'
                                            ? `Błąd: ${file.error_message || 'Nieznany'}`
                                            : `Status: ${file.status || 'Nieznany'}`}
                                    </span>
                                </p>

                                {file.status === 'completed' && file.file_id && (
                                    <div className="space-y-1 text-sm text-gray-700">
                                        <p>
                                            <strong>Szczegóły:</strong>{' '}
                                            <Link
                                                to={`/processed_files/${file.file_id}/details`}
                                                className="text-gray-900 underline decoration-transparent hover:decoration-current"
                                            >
                                                Zobacz szczegóły pliku
                                            </Link>
                                        </p>
                                        {file.transcription_url && (
                                            <p>
                                                <strong>Transkrypcja:</strong>{' '}
                                                <Link
                                                    to={`${file.transcription_url}?token=${token}`}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="text-gray-900 underline decoration-transparent hover:decoration-current"
                                                >
                                                    Zobacz
                                                </Link>{' '}
                                                |{' '}
                                                <a
                                                    href={`${file.transcription_download_url}?token=${token}`}
                                                    download
                                                    className="text-gray-900 underline decoration-transparent hover:decoration-current"
                                                >
                                                    Pobierz
                                                </a>
                                            </p>
                                        )}
                                        {file.minutes_html_url && (
                                            <p>
                                                <strong>Protokoły:</strong>{' '}
                                                <Link
                                                    to={`${file.minutes_html_url}&token=${token}`}
                                                    target="_blank"
                                                    rel="noopener noreferrer"
                                                    className="text-gray-900 underline decoration-transparent hover:decoration-current"
                                                >
                                                    Zobacz HTML
                                                </Link>{' '}
                                                |{' '}
                                                <a
                                                    href={`${file.minutes_download_url}?token=${token}`}
                                                    download
                                                    className="text-gray-900 underline decoration-transparent hover:decoration-current"
                                                >
                                                    Pobierz
                                                </a>
                                            </p>
                                        )}
                                    </div>
                                )}
                                {file.status === 'error' && file.error_message && (
                                    <p className="text-sm text-red-600">Błąd: {file.error_message}</p>
                                )}
                            </div>
                        ))}
                    </div>
                </section>

                <div>
                    <Link
                        to="/upload"
                        className="inline-flex items-center rounded-md bg-gray-900 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-gray-800"
                    >
                        Prześlij nową partię
                    </Link>
                </div>
            </>
        );
    }

    return (
        <PageLayout title={pageTitle} description={pageDescription}>
            {content}
        </PageLayout>
    );
}

export default ProcessedBatchDetailsPage;
