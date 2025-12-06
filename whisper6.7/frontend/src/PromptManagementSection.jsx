import React, { useState, useEffect, useRef, useCallback, useImperativeHandle, forwardRef } from 'react';
import { useAuth, useAuthorizedFetch } from './AuthContext';

const PromptManagementSection = forwardRef(({ isAuthenticated, updateStatus }, ref) => {
    const authorizedFetch = useAuthorizedFetch();

    useImperativeHandle(ref, () => ({
        generateCustomPromptJson
    }));

    const [customPrompts, setCustomPrompts] = useState([{ title: '', query: '' }]);
    const [savedPromptSets, setSavedPromptSets] = useState([]);
    const [readOnlyPromptSets, setReadOnlyPromptSets] = useState([]);
    const [selectedPromptSet, setSelectedPromptSet] = useState('');
    const uploadJsonPromptFileRef = useRef(null);
    const savedPromptSetsSelectRef = useRef(null);
    const promptSetsRef = useRef({});

    // Function to generate custom prompts JSON
    const generateCustomPromptJson = useCallback(() => {
        const prompts = {};
        customPrompts.forEach(section => {
            if (section.title.trim() && section.query.trim()) {
                prompts[section.title.trim()] = section.query.trim();
            }
        });
        return Object.keys(prompts).length > 0 ? JSON.stringify(prompts) : null;
    }, [customPrompts]);

    // Function to add a new prompt section
    const addPromptSection = useCallback((title = '', query = '') => {
        setCustomPrompts(prev => [...prev, { title, query }]);
    }, []);

    // Function to handle removing a prompt section
    const handleRemovePromptSection = useCallback((index) => {
        setCustomPrompts(prev => prev.filter((_, i) => i !== index));
    }, []);

    const loadPromptSets = useCallback(async () => {
        if (!isAuthenticated) {
            promptSetsRef.current = {};
            setSavedPromptSets([]);
            setReadOnlyPromptSets([]);
            setSelectedPromptSet('');
            return;
        }

        try {
            const response = await authorizedFetch('/prompts/list');
            if (response.ok) {
                const payload = await response.json();
                const promptsMap = payload?.prompts || {};
                promptSetsRef.current = promptsMap;
                setSavedPromptSets(Object.keys(promptsMap));
                setReadOnlyPromptSets(Array.isArray(payload?.readonly) ? payload.readonly : []);
                setSelectedPromptSet((prev) => (prev && promptsMap[prev] ? prev : ''));
            } else {
                const errorText = await response.text();
                updateStatus(`Błąd ładowania zestawów promptów: ${errorText}`, 'error');
            }
        } catch (error) {
            updateStatus(`Wystąpił błąd sieci podczas ładowania zestawów promptów: ${error.message}`, 'error');
        }
    }, [authorizedFetch, isAuthenticated, updateStatus]);

    useEffect(() => {
        loadPromptSets();
    }, [loadPromptSets]);

    const handleLoadPromptSet = () => {
        const selectedName = selectedPromptSet;
        if (!selectedName) {
            alert('Proszę wybrać zestaw promptów do załadowania.');
            return;
        }
        const promptsToLoad = promptSetsRef.current[selectedName];
        if (!promptsToLoad) {
            updateStatus(`Zestaw promptów '${selectedName}' nie znaleziono.`, 'error');
            return;
        }

        const mappedPrompts = Object.entries(promptsToLoad).map(([title, query]) => ({
            title,
            query
        }));
        setCustomPrompts(mappedPrompts.length > 0 ? mappedPrompts : [{ title: '', query: '' }]);
        updateStatus(`Zestaw promptów '${selectedName}' załadowany.`, 'info');
    };

    const handleSaveCurrentPromptSet = async () => {
        const promptName = prompt('Wprowadź nazwę dla bieżącego zestawu promptów:');
        if (!promptName) return;

        const prompts = {};
        customPrompts.forEach((section) => {
            const title = section.title.trim();
            const query = section.query.trim();
            if (title && query) {
                prompts[title] = query;
            }
        });

        if (Object.keys(prompts).length === 0) {
            alert('Nie można zapisać pustego zestawu promptów. Proszę dodać przynajmniej jeden prompt.');
            return;
        }

        try {
            const response = await authorizedFetch('/prompts/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: promptName, prompts }),
            });
            if (response.ok) {
                const result = await response.json().catch(() => ({}));
                updateStatus(result.message || `Zestaw promptów '${promptName}' został zapisany.`, 'success');
                setCustomPrompts([{ title: '', query: '' }]);
                if (uploadJsonPromptFileRef.current) {
                    uploadJsonPromptFileRef.current.value = '';
                }
                await loadPromptSets();
                setSelectedPromptSet(promptName);
            } else {
                const errorText = await response.text();
                updateStatus(`Błąd zapisu zestawu promptów: ${errorText}`, 'error');
            }
        } catch (error) {
            updateStatus(`Wystąpił błąd sieci podczas zapisu zestawu promptów: ${error.message}`, 'error');
        }
    };

    const handleDeletePromptSet = async () => {
        const selectedName = selectedPromptSet;
        if (!selectedName) {
            alert('Proszę wybrać zestaw promptów do usunięcia.');
            return;
        }
        if (readOnlyPromptSets.includes(selectedName)) {
            updateStatus('Zestawy domyślne są tylko do odczytu i nie można ich usuwać.', 'warning');
            return;
        }
        if (!confirm(`Czy na pewno chcesz usunąć zestaw promptów '${selectedName}'?`)) {
            return;
        }

        try {
            const response = await authorizedFetch(`/prompts/delete/${encodeURIComponent(selectedName)}`, {
                method: 'DELETE',
            });
            if (response.ok) {
                const result = await response.json().catch(() => ({}));
                updateStatus(result.message || `Zestaw promptów '${selectedName}' został usunięty.`, 'success');
                setSelectedPromptSet('');
                setCustomPrompts([{ title: '', query: '' }]);
                await loadPromptSets();
            } else {
                const errorText = await response.text();
                updateStatus(`Błąd usuwania zestawu promptów: ${errorText}`, 'error');
            }
        } catch (error) {
            updateStatus(`Wystąpił błąd sieci podczas usuwania zestawu promptów: ${error.message}`, 'error');
        }
    };

    const handleUploadJsonPromptFile = (event) => {
        const file = event.target.files[0];
        if (!file) {
            return;
        }

        const reader = new FileReader();
        reader.onload = function(e) {
            try {
                const promptsFromFile = JSON.parse(e.target.result);
                if (typeof promptsFromFile === 'object' && promptsFromFile !== null) {
                    const promptEntries = Object.entries(promptsFromFile).map(([title, query]) => ({
                        title,
                        query
                    }));
                    setCustomPrompts(promptEntries.length > 0 ? promptEntries : [{ title: '', query: '' }]);
                    updateStatus('Zestaw promptów wczytany z pliku JSON.', 'success');
                } else {
                    updateStatus('Nieprawidłowy format pliku JSON. Oczekiwano obiektu.', 'error');
                }
            } catch (error) {
                updateStatus(`Błąd parsowania pliku JSON: ${error.message}`, 'error');
            }
        };
        reader.readAsText(file);
    };

    const hasSelectedPrompt = Boolean(selectedPromptSet);
    const isSelectedPromptReadOnly = hasSelectedPrompt && readOnlyPromptSets.includes(selectedPromptSet);

    return (
        <section className="space-y-6 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
            <div className="space-y-1">
                <h3 className="text-lg font-semibold text-gray-800">
                    Definiuj własne sekcje minut spotkania <span className="text-sm font-normal text-gray-500">(opcjonalnie)</span>
                </h3>
                <p className="text-sm text-gray-600">
                    Dodaj pytania, które chcesz zadać AI, aby spersonalizować podsumowanie spotkania.
                </p>
            </div>

            <div className="space-y-4">
                {customPrompts.map((prompt, index) => (
                    <div
                        key={index}
                        className="space-y-3 rounded-lg border border-gray-200 bg-gray-50 p-4 shadow-sm"
                    >
                        <div className="space-y-1">
                            <label className="text-xs font-semibold uppercase tracking-wide text-gray-500">
                                Sekcja #{index + 1}
                            </label>
                            <input
                                type="text"
                                className="block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-900/20"
                                placeholder="Nazwa sekcji (np. 'Główne Tematy')"
                                value={prompt.title}
                                onChange={(e) => {
                                    const newPrompts = [...customPrompts];
                                    newPrompts[index].title = e.target.value;
                                    setCustomPrompts(newPrompts);
                                }}
                            />
                        </div>
                        <div className="space-y-1">
                            <label className="text-xs font-semibold uppercase tracking-wide text-gray-500">
                                Pytanie
                            </label>
                            <textarea
                                className="block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-900/20"
                                rows="3"
                                placeholder="Pytanie do AI (np. 'Jakie były główne tematy spotkania?')"
                                value={prompt.query}
                                onChange={(e) => {
                                    const newPrompts = [...customPrompts];
                                    newPrompts[index].query = e.target.value;
                                    setCustomPrompts(newPrompts);
                                }}
                            ></textarea>
                        </div>
                        <div className="flex justify-end">
                            <button
                                type="button"
                                className="inline-flex items-center rounded-md bg-red-500 px-3 py-1.5 text-xs font-semibold text-white shadow-sm transition hover:bg-red-600"
                                onClick={() => handleRemovePromptSection(index)}
                            >
                                Usuń sekcję
                            </button>
                        </div>
                    </div>
                ))}
            </div>

            <button
                type="button"
                className="inline-flex items-center rounded-md border border-dashed border-gray-600 bg-white px-4 py-2 text-sm font-semibold text-gray-900 shadow-md transition hover:bg-gray-100 hover:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-400"
                onClick={() => addPromptSection()}
            >
                Dodaj sekcję promptu
            </button>

            <div className="space-y-4 rounded-lg border border-gray-200 bg-gray-50 p-4 shadow-sm">
                <div className="space-y-1">
                    <label htmlFor="savedPromptSetsSelect" className="text-sm font-semibold text-gray-700">
                        Zapisane zestawy promptów
                    </label>
                    <select
                        id="savedPromptSetsSelect"
                        ref={savedPromptSetsSelectRef}
                        className="block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-900/20"
                        value={selectedPromptSet}
                        onChange={(e) => setSelectedPromptSet(e.target.value)}
                    >
                        <option value="">-- Wybierz zestaw --</option>
                        {savedPromptSets.map((name) => (
                            <option key={name} value={name}>
                                {name}{readOnlyPromptSets.includes(name) ? ' (domyślny)' : ''}
                            </option>
                        ))}
                    </select>
                </div>

                <div className="flex flex-wrap gap-2">
                    <button
                        type="button"
                        className="inline-flex items-center rounded-md border border-gray-700 bg-white px-4 py-2 text-sm font-semibold text-gray-900 shadow-md transition hover:bg-gray-100 hover:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-400 disabled:cursor-not-allowed disabled:opacity-50"
                        onClick={handleLoadPromptSet}
                        disabled={!hasSelectedPrompt}
                    >
                        Załaduj wybrany
                    </button>
                    <button
                        type="button"
                        className="inline-flex items-center rounded-md border border-gray-700 bg-white px-4 py-2 text-sm font-semibold text-gray-900 shadow-md transition hover:bg-gray-100 hover:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-400"
                        onClick={handleSaveCurrentPromptSet}
                    >
                        Zapisz bieżący
                    </button>
                    <button
                        type="button"
                        className="inline-flex items-center rounded-md bg-red-500 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-red-600 disabled:cursor-not-allowed disabled:opacity-50"
                        onClick={handleDeletePromptSet}
                        disabled={!hasSelectedPrompt || isSelectedPromptReadOnly}
                    >
                        Usuń wybrany
                    </button>
                </div>

                <div className="space-y-2 rounded-md border border-dashed border-gray-300 bg-white p-3">
                    <label
                        htmlFor="uploadJsonPromptFile"
                        className="text-sm font-medium text-gray-700"
                    >
                        Wczytaj zestaw promptów z pliku JSON
                    </label>
                    <input
                        type="file"
                        id="uploadJsonPromptFile"
                        accept=".json"
                        ref={uploadJsonPromptFileRef}
                        onChange={handleUploadJsonPromptFile}
                        className="block w-full text-sm text-gray-600 file:mr-4 file:rounded-md file:border-0 file:bg-gray-200 file:px-4 file:py-2 file:text-gray-900 hover:file:bg-gray-300 focus:outline-none"
                    />
                    <p className="text-xs text-gray-500">
                        Plik powinien zawierać obiekt JSON z parą „nazwa sekcji” → „pytanie”.
                    </p>
                </div>
            </div>

            <input
                type="hidden"
                id="generatedCustomPrompt"
                name="generated_custom_prompt"
                value={generateCustomPromptJson() || ''}
            />
        </section>
    );
});

export default PromptManagementSection;
