import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { useAuthorizedFetch } from './AuthContext';
import PageLayout from './PageLayout';

function TranscriptPage() {
    const { fileId } = useParams();
    const authorizedFetch = useAuthorizedFetch();
    const [transcriptContent, setTranscriptContent] = useState('');
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    useEffect(() => {
        const fetchTranscript = async () => {
            try {
                const response = await authorizedFetch(`/processed_files/${fileId}/transcript`);
                if (!response.ok) {
                    throw new Error(`Błąd HTTP: ${response.status} ${response.statusText}`);
                }
                const content = await response.text(); // Transcript is HTML response
                setTranscriptContent(content);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };

        fetchTranscript();
    }, [fileId, authorizedFetch]);

    const pageTitle = `Transkrypcja pliku`;
    const pageDescription = `Identyfikator: ${fileId}`;

    let content;
    if (loading) {
        content = (
            <div className="rounded-xl border border-gray-200 bg-white p-6 text-sm text-gray-600 shadow-sm">
                Ładowanie transkrypcji...
            </div>
        );
    } else if (error) {
        content = (
            <div className="rounded-xl border border-red-200 bg-red-50 p-6 text-sm text-red-700 shadow-sm">
                Błąd: {error}
            </div>
        );
    } else if (!transcriptContent) {
        content = (
            <div className="rounded-xl border border-gray-200 bg-white p-6 text-sm text-gray-600 shadow-sm">
                Brak transkrypcji.
            </div>
        );
    } else {
        content = (
            <div className="rounded-xl border border-gray-200 bg-white p-6 shadow-sm">
                <div
                    className="transcript-html max-w-full overflow-x-auto text-sm text-gray-800"
                    dangerouslySetInnerHTML={{ __html: transcriptContent }}
                />
            </div>
        );
    }

    return (
        <PageLayout title={pageTitle} description={pageDescription}>
            {content}
        </PageLayout>
    );
}

export default TranscriptPage;
