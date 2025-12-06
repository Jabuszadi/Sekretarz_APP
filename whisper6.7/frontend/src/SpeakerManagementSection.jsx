import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useAuth, useAuthorizedFetch } from './AuthContext';
import { Link } from 'react-router-dom';

const SpeakerManagementSection = ({ updateStatus }) => {
    const { isAuthenticated } = useAuth();
    const authorizedFetch = useAuthorizedFetch();

    const [enrolledSpeakers, setEnrolledSpeakers] = useState([]);
    const [speakerName, setSpeakerName] = useState('');
    const speakerAudioFileInputRef = useRef(null);

    const loadEnrolledSpeakers = useCallback(async () => {
        if (!isAuthenticated) {
            console.log("SpeakerManagementSection: Not authenticated, skipping loadEnrolledSpeakers.");
            setEnrolledSpeakers([]);
            return;
        }
        try {
            const response = await authorizedFetch('/get_enrolled_speakers/');
            if (response.ok) {
                const speakers = await response.json();
                // console.log('DEBUG: Speakers received from API:', speakers);
                setEnrolledSpeakers(speakers);
            } else {
                const errorData = await response.json();
                updateStatus(`Błąd ładowania mówców: ${errorData.detail || response.statusText}`, 'error');
                setEnrolledSpeakers([]);
            }
        } catch (error) {
            updateStatus(`Wystąpił błąd sieci podczas ładowania zarejestrowanych mówców: ${error.message}`, 'error');
            setEnrolledSpeakers([]);
        }
    }, [updateStatus, authorizedFetch, isAuthenticated]);

    useEffect(() => {
        console.log("SpeakerManagementSection useEffect - isAuthenticated:", isAuthenticated);
        if (isAuthenticated) {
            loadEnrolledSpeakers();
        }
    }, [isAuthenticated, loadEnrolledSpeakers]);

    const handleSpeakerEnrollmentSubmit = async (e) => {
        e.preventDefault();
        const speakerNameValue = speakerName.trim();
        const audioFile = speakerAudioFileInputRef.current.files[0];

        if (!speakerNameValue) {
            updateStatus('Proszę podać nazwę mówcy.', 'error');
            return;
        }

        if (!audioFile) {
            updateStatus('Plik audio jest wymagany.', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('speaker_name_form', speakerNameValue);
        formData.append('audio_file', audioFile);

        updateStatus(`Zapisywanie mówcy '${speakerNameValue}'...`, 'info');

        try {
            const response = await authorizedFetch('/enroll_speaker_direct/', {
                method: 'POST',
                body: formData,
            });

            if (response.ok) {
                const result = await response.json();
                updateStatus(result.message, 'success');
                setSpeakerName('');
                speakerAudioFileInputRef.current.value = '';
                loadEnrolledSpeakers();
            } else {
                const errorData = await response.json();
                updateStatus(`Błąd zapisu mówcy: ${errorData.detail || response.statusText}`, 'error');
            }
        } catch (error) {
            updateStatus(`Wystąpił błąd sieci podczas zapisu mówcy: ${error.message}`, 'error');
        }
    };

    const handleDeleteSpeaker = async (speaker) => {
        if (!confirm(`Czy na pewno chcesz usunąć mówcę '${speaker}'?`)) {
            return;
        }

        try {
            const deleteResponse = await authorizedFetch(`/delete_speaker/${speaker}`, {
                method: 'DELETE',
            });
            if (deleteResponse.ok) {
                updateStatus(`Mówca '${speaker}' został usunięty.`, 'success');
                loadEnrolledSpeakers();
            } else {
                const errorData = await deleteResponse.json();
                updateStatus(`Błąd usuwania mówcy: ${errorData.detail || deleteResponse.statusText}`, 'error');
            }
        } catch (error) {
            updateStatus(`Wystąpił błąd sieci podczas usuwania mówcy: ${error.message}`, 'error');
        }
    };

    const disabledClass = !isAuthenticated ? 'opacity-50 pointer-events-none' : '';

    return (
        <div id="recognizer" className="space-y-6">
            <div>
                <h2 className="text-xl font-semibold text-gray-800">Zarządzanie Mówcami</h2>
                <p className="text-sm text-gray-600 mt-1">
                    Dodawaj próbki głosu, zarządzaj istniejącymi mówcami i aktualizuj bazę rozpoznawania.
                </p>
            </div>

            {!isAuthenticated && (
                <div className="text-red-600 bg-red-100 border border-red-200 rounded-md px-4 py-3">
                    Musisz być zalogowany, aby zarządzać mówcami.&nbsp;
                    <Link to="/login" className="text-red-700 underline font-medium">
                        Zaloguj się
                    </Link>
                </div>
            )}

            <section
                className={`bg-gray-50 border border-gray-200 rounded-lg shadow-sm p-5 transition-opacity ${disabledClass}`}
            >
                <h3 className="text-lg font-semibold text-gray-800 mb-4">Bezpośredni zapis mówcy</h3>
                <form
                    id="directSpeakerEnrollmentForm"
                    onSubmit={handleSpeakerEnrollmentSubmit}
                    className="space-y-4"
                >
                    <div className="space-y-1">
                        <label
                            htmlFor="speakerNameInput"
                            className="block text-sm font-medium text-gray-700"
                        >
                            Nazwa mówcy
                        </label>
                        <input
                            type="text"
                            id="speakerNameInput"
                            placeholder="Wprowadź nazwę mówcy (np. 'Jan Kowalski')"
                            className="block w-full rounded-md border border-gray-300 px-3 py-2 text-sm focus:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-900/30 bg-white"
                            value={speakerName}
                            onChange={(e) => setSpeakerName(e.target.value)}
                            disabled={!isAuthenticated}
                        />
                    </div>

                    <div className="space-y-1">
                        <label
                            htmlFor="speakerAudioFileInput"
                            className="block text-sm font-medium text-gray-700"
                        >
                            Plik audio mówcy (krótka próbka głosu)
                        </label>
                        <input
                            type="file"
                            id="speakerAudioFileInput"
                            accept=".wav,.mp3,.flac,.aac,.ogg,.m4a"
                            required
                            ref={speakerAudioFileInputRef}
                            className="block w-full text-sm text-gray-700 file:mr-4 file:rounded-md file:border-0 file:bg-gray-200 file:px-4 file:py-2 file:text-gray-900 hover:file:bg-gray-300 focus:outline-none"
                            disabled={!isAuthenticated}
                        />
                        <p className="text-xs text-gray-500">
                            Wgraj plik audio o długości do 30 sekund – najlepiej w formacie WAV/MP3.
                        </p>
                    </div>

                    <div className="flex justify-end">
                        <button
                            type="submit"
                            id="enrollSpeakerDirectButton"
                            disabled={!isAuthenticated}
                            className="inline-flex items-center rounded-md border border-gray-400 bg-white px-4 py-2 text-sm font-semibold text-gray-900 shadow-sm transition hover:bg-gray-100 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            Zapisz mówcę
                        </button>
                    </div>
                </form>
            </section>

            <section
                className={`bg-white border border-gray-200 rounded-lg shadow-sm p-5 transition-opacity ${disabledClass}`}
            >
                <div className="flex items-center justify-between mb-4">
                    <div>
                        <h3 className="text-lg font-semibold text-gray-800">Zarejestrowani mówcy</h3>
                        <p className="text-sm text-gray-600">
                            Zarządzaj listą mówców dostępnych do rozpoznawania.
                        </p>
                    </div>
                    <button
                        type="button"
                        id="refreshEnrolledSpeakersButton"
                        onClick={loadEnrolledSpeakers}
                        disabled={!isAuthenticated}
                        className="inline-flex items-center rounded-md border border-gray-700 bg-white px-4 py-2 text-sm font-semibold text-gray-900 shadow-md transition hover:bg-gray-100 hover:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-400 disabled:cursor-not-allowed disabled:opacity-50"
                        className="inline-flex items-center rounded-md border border-gray-700 bg-white px-4 py-2 text-sm font-semibold text-gray-900 shadow-md transition hover:bg-gray-100 hover:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-400 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        Odśwież listę
                    </button>
                </div>

                <ul
                    id="enrolledSpeakersList"
                    className="divide-y divide-gray-200 rounded-md border border-gray-200 bg-gray-50"
                >
                    {enrolledSpeakers.length > 0 ? (
                        enrolledSpeakers.map((speaker) => (
                            <li
                                key={speaker}
                                className="flex items-center justify-between px-4 py-3 text-sm text-gray-700"
                            >
                                <span className="font-medium text-gray-800">{speaker}</span>
                                <button
                                    onClick={() => handleDeleteSpeaker(speaker)}
                                    className="inline-flex items-center rounded-md border border-red-600 bg-white px-3 py-1.5 text-xs font-semibold text-red-700 shadow-sm transition hover:bg-red-50 hover:border-red-700 focus:outline-none focus:ring-2 focus:ring-red-400"
                                >
                                    Usuń
                                </button>
                            </li>
                        ))
                    ) : (
                        <li className="px-4 py-4 text-sm text-gray-600">
                            Brak zarejestrowanych mówców.
                        </li>
                    )}
                </ul>
            </section>
        </div>
    );
};

export default SpeakerManagementSection;
