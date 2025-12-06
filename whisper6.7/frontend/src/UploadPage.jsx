import React, { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate, Link } from 'react-router-dom'; // Import useNavigate
import { useAuth, useAuthorizedFetch } from './AuthContext'; // Dodaj useAuthorizedFetch
import Select from 'react-select';
import PromptManagementSection from './PromptManagementSection'; // Importuj nowy komponent
import SpeakerManagementSection from './SpeakerManagementSection'; // Importuj nowy komponent
import { useApiTokens } from './ApiTokensContext';
import PageLayout from './PageLayout';

const SESSION_STORAGE_KEY = 'sekretarzUploadState';

const ASSEMBLYAI_ALLOWED_MODELS = new Set(['best', 'nano', 'slam-1', 'universal']);

const DEFAULT_TRANSCRIPTION_PROVIDERS = [
    {
        provider_id: 'gpt',
        label: 'OpenAI Whisper',
        default_model: 'normal',
        models: [
            { id: 'mini', label: 'Mini (gpt-4o-mini-transcribe)' },
            { id: 'normal', label: 'Normal (gpt-4o-transcribe)' },
        ],
    },
    {
        provider_id: 'gemini',
        label: 'Gemini',
        default_model: 'gemini-2.5-flash',
        models: [
            { id: 'gemini-2.5-flash', label: 'Gemini 2.5 Flash' },
            { id: 'gemini-2.5-pro', label: 'Gemini 2.5 Pro' },
        ],
    },
    {
        provider_id: 'assemblyai',
        label: 'AssemblyAI',
        default_model: 'best',
        models: [
            { id: 'best', label: 'Best' },
            { id: 'nano', label: 'Nano' },
            { id: 'slam-1', label: 'SLAM-1' },
            { id: 'universal', label: 'Universal' },
        ],
    },
];

const GPT_LEGACY_MODEL_REDIRECTS = {
    tiny: 'mini',
    base: 'mini',
    small: 'mini',
    medium: 'normal',
    large: 'normal',
    'large-v1': 'normal',
    'large-v2': 'normal',
    'large-v3': 'normal',
};

const normalizeProviderValue = (value) => {
    if (typeof value !== 'string') {
        return '';
    }
    return value.trim().toLowerCase();
};

const normalizeModelValue = (providerId, value) => {
    if (typeof value !== 'string') {
        return '';
    }
    const trimmed = value.trim();
    if (!trimmed) {
        return '';
    }
    const lower = trimmed.toLowerCase();
    if (providerId === 'gpt') {
        if (lower === 'mini' || lower === 'normal') {
            return lower;
        }
        const redirect = GPT_LEGACY_MODEL_REDIRECTS[lower];
        if (redirect) {
            return redirect;
        }
    }
    if (providerId === 'assemblyai') {
        if (ASSEMBLYAI_ALLOWED_MODELS.has(lower)) {
            return lower;
        }
        return DEFAULT_TRANSCRIPTION_PROVIDERS.find((entry) => entry.provider_id === 'assemblyai')?.default_model || 'best';
    }
    return trimmed;
};

const FALLBACK_TRANSCRIPTION_PROVIDER = DEFAULT_TRANSCRIPTION_PROVIDERS[0].provider_id;
const FALLBACK_TRANSCRIPTION_MODEL =
    DEFAULT_TRANSCRIPTION_PROVIDERS[0].default_model || DEFAULT_TRANSCRIPTION_PROVIDERS[0].models[0].id;

const STATUS_STYLE_MAP = {
    info: 'bg-gray-100 text-gray-900 border border-gray-300',
    success: 'bg-green-50 text-green-700 border border-green-200',
    warning: 'bg-yellow-50 text-yellow-700 border border-yellow-200',
    error: 'bg-red-50 text-red-700 border border-red-200',
    default: 'bg-gray-50 text-gray-700 border border-gray-200',
};

function UploadPage() {
    const navigate = useNavigate();
    const { token, isAuthenticated } = useAuth(); // Pobierz token i status autoryzacji
    const authorizedFetch = useAuthorizedFetch(); // Użyj nowego hooka
    const { tokens: sessionTokens } = useApiTokens();
    const [initialCachedState] = useState(() => {
        if (typeof window !== 'undefined' && window.sessionStorage) {
            const storedValue = window.sessionStorage.getItem(SESSION_STORAGE_KEY);
            if (storedValue) {
                try {
                    return JSON.parse(storedValue);
                } catch (parseError) {
                    console.warn('SessionStorage parse failed:', parseError);
                }
            }
        }
        return null;
    });
    const initialCachedProvider = initialCachedState?.transcriptionProvider ?? FALLBACK_TRANSCRIPTION_PROVIDER;
    const initialCachedModel = initialCachedState?.transcriptionModel ?? initialCachedState?.whisperModelSize;
    const [activeTab, setActiveTab] = useState('mainProcessing'); // NOWY STAN: do zarządzania aktywną zakładką
    const [transcriptionProviders, setTranscriptionProviders] = useState(DEFAULT_TRANSCRIPTION_PROVIDERS);
    const initialProviderValue = (() => {
        const normalized = normalizeProviderValue(initialCachedProvider);
        if (normalized && DEFAULT_TRANSCRIPTION_PROVIDERS.some((entry) => entry.provider_id === normalized)) {
            return normalized;
        }
        return FALLBACK_TRANSCRIPTION_PROVIDER;
    })();
    const [selectedTranscriptionProvider, setSelectedTranscriptionProvider] = useState(initialProviderValue);
    const initialProviderEntry =
        DEFAULT_TRANSCRIPTION_PROVIDERS.find((entry) => entry.provider_id === initialProviderValue) ||
        DEFAULT_TRANSCRIPTION_PROVIDERS[0];
    const initialModelValue = (() => {
        const normalized = normalizeModelValue(initialProviderValue, initialCachedModel);
        if (normalized && initialProviderEntry?.models?.some((model) => model.id === normalized)) {
            return normalized;
        }
        return initialProviderEntry?.default_model || initialProviderEntry?.models?.[0]?.id || FALLBACK_TRANSCRIPTION_MODEL;
    })();
    const [transcriptionModels, setTranscriptionModels] = useState(initialProviderEntry?.models || []);
    const [isLoadingTranscriptionOptions, setIsLoadingTranscriptionOptions] = useState(false);
    const [selectedTranscriptionModel, setSelectedTranscriptionModel] = useState(initialModelValue);

    // State variables for various UI elements, mimicking the original JS logic
    const [statusMessage, setStatusMessage] = useState(initialCachedState?.statusMessage ?? 'Oczekiwanie na plik...');
    const [statusType, setStatusType] = useState(initialCachedState?.statusType ?? 'info');
    const [batchStatusMessage, setBatchStatusMessage] = useState(initialCachedState?.batchStatusMessage ?? '');
    const [batchProgressPercentage, setBatchProgressPercentage] = useState(initialCachedState?.batchProgressPercentage ?? 0); // NOWY STAN
    const [completedFiles, setCompletedFiles] = useState(initialCachedState?.completedFiles ?? 0); // NOWY STAN
    const [totalFiles, setTotalFiles] = useState(initialCachedState?.totalFiles ?? 0); // NOWY STAN
    const [currentFileName, setCurrentFileName] = useState(initialCachedState?.currentFileName ?? '');
    const [currentFileProgressPercentage, setCurrentFileProgressPercentage] = useState(initialCachedState?.currentFileProgressPercentage ?? 0); // NOWY STAN
    const [currentFileStatusMessage, setCurrentFileStatusMessage] = useState(initialCachedState?.currentFileStatusMessage ?? ''); // NOWY STAN
    const [currentFileWhisperProgressPercentage, setCurrentFileWhisperProgressPercentage] = useState(initialCachedState?.currentFileWhisperProgressPercentage ?? 0);
    const [currentFileWhisperStatusMessage, setCurrentFileWhisperStatusMessage] = useState(initialCachedState?.currentFileWhisperStatusMessage ?? '');
    const [currentFileMinutesProgressPercentage, setCurrentFileMinutesProgressPercentage] = useState(initialCachedState?.currentFileMinutesProgressPercentage ?? 0);
    const [currentFileMinutesStatusMessage, setCurrentFileMinutesStatusMessage] = useState(initialCachedState?.currentFileMinutesStatusMessage ?? '');
    const [currentFileMinutesSection, setCurrentFileMinutesSection] = useState(initialCachedState?.currentFileMinutesSection ?? '');
    const [sendButtonDisabled, setSendButtonDisabled] = useState(initialCachedState?.sendButtonDisabled ?? false);
    const [advancedOptionsVisible, setAdvancedOptionsVisible] = useState(false);
    const [isFileSelected, setIsFileSelected] = useState(false); // NOWY STAN
    const [processedResults, setProcessedResults] = useState(initialCachedState?.processedResults ?? []); // NEW: State for processed file results
    const [selectedFileNames, setSelectedFileNames] = useState(initialCachedState?.selectedFileNames ?? []);
    const [isProcessingActive, setIsProcessingActive] = useState(initialCachedState?.isProcessingActive ?? false);
    const [currentBatchJobId, setCurrentBatchJobId] = useState(initialCachedState?.batchJobId ?? null);
    const [fileNameMapState, setFileNameMapState] = useState(initialCachedState?.fileNameMap ?? {});
    const fileInputRef = useRef(null);
    const providerSelectRef = useRef(null);
    const transcriptionModelSelectRef = useRef(null);
    const chunkDurationRef = useRef(null);
    const chunkOverlapRef = useRef(null);
    const currentFileNameRef = useRef(initialCachedState?.currentFileName ?? '');
    const selectedTranscriptionProviderRef = useRef(selectedTranscriptionProvider);
    const selectedTranscriptionModelRef = useRef(selectedTranscriptionModel);
    const eventSourceRef = useRef(null);
    const cachedStateRef = useRef(initialCachedState);
    const resumeRetryTimeoutRef = useRef(null);
    const fileNameMapRef = useRef(fileNameMapState);

    // Stwórz ref dla PromptManagementSection
    const promptManagementRef = useRef(null);
    const hasResumedRef = useRef(false);

    useEffect(() => {
        currentFileNameRef.current = currentFileName;
    }, [currentFileName]);

    useEffect(() => {
        fileNameMapRef.current = fileNameMapState;
    }, [fileNameMapState]);

    useEffect(() => {
        selectedTranscriptionProviderRef.current = selectedTranscriptionProvider;
        if (providerSelectRef.current && typeof selectedTranscriptionProvider === 'string') {
            providerSelectRef.current.value = selectedTranscriptionProvider;
        }
    }, [selectedTranscriptionProvider]);

    useEffect(() => {
        selectedTranscriptionModelRef.current = selectedTranscriptionModel;
        if (transcriptionModelSelectRef.current && typeof selectedTranscriptionModel === 'string') {
            transcriptionModelSelectRef.current.value = selectedTranscriptionModel;
        }
    }, [selectedTranscriptionModel]);

    const extractFileDisplayName = useCallback((payload) => {
        if (!payload) {
            return '';
        }

        const map = fileNameMapRef.current || {};
        const candidateKeys = [
            payload.file_job_id,
            payload.file_id !== undefined && payload.file_id !== null ? String(payload.file_id) : null,
            payload?.batch_progress?.file_job_id,
        ].filter(Boolean);

        for (const key of candidateKeys) {
            const mapped = map[key];
            if (mapped && mapped.trim().length > 0) {
                return mapped.trim();
            }
        }

        const candidates = [
            payload.original_filename,
            payload.display_name,
            payload.name,
            payload.current_file_name,
            payload.file_name,
            payload.file,
            payload?.batch_progress?.current_file_name,
            payload.transcription_filename
                ? payload.transcription_filename.split(/[\\/]/).pop()
                : '',
            payload.minutes_filename
                ? payload.minutes_filename.split(/[\\/]/).pop()
                : '',
        ];

        const found = candidates.find(
            (value) => typeof value === 'string' && value.trim().length > 0
        );

        return found ? found.trim() : '';
    }, []);

    const persistState = useCallback((partialState) => {
        const updatedState = {
            ...(cachedStateRef.current || {}),
            ...partialState,
        };
        cachedStateRef.current = updatedState;

        if (typeof window !== 'undefined' && window.sessionStorage) {
            try {
                window.sessionStorage.setItem(SESSION_STORAGE_KEY, JSON.stringify(updatedState));
            } catch (storageError) {
                console.warn('SessionStorage persist failed:', storageError);
            }
        }
    }, []);

    const handleTranscriptionProviderChange = useCallback(
        (event) => {
            const rawValue = event?.target?.value ?? '';
            const normalizedProvider = normalizeProviderValue(rawValue) || FALLBACK_TRANSCRIPTION_PROVIDER;
            const providerEntry =
                transcriptionProviders.find((provider) => provider.provider_id === normalizedProvider) ||
                transcriptionProviders[0] ||
                DEFAULT_TRANSCRIPTION_PROVIDERS[0];
            const providerId = providerEntry?.provider_id || FALLBACK_TRANSCRIPTION_PROVIDER;
            const providerModels = providerEntry?.models || [];
            setSelectedTranscriptionProvider(providerId);
            setTranscriptionModels(providerModels);

            const defaultModelCandidate = providerEntry?.default_model || providerModels[0]?.id;
            const normalizedModel =
                normalizeModelValue(providerId, selectedTranscriptionModelRef.current) ||
                normalizeModelValue(providerId, defaultModelCandidate) ||
                providerModels[0]?.id ||
                FALLBACK_TRANSCRIPTION_MODEL;
            setSelectedTranscriptionModel(normalizedModel);
            persistState({
                transcriptionProvider: providerId,
                transcriptionModel: normalizedModel,
            });
        },
        [persistState, transcriptionProviders],
    );

    const handleTranscriptionModelChange = useCallback(
        (event) => {
            const rawValue = event?.target?.value ?? '';
            const normalizedValue = normalizeModelValue(selectedTranscriptionProvider, rawValue);
            if (normalizedValue) {
                setSelectedTranscriptionModel(normalizedValue);
                persistState({
                    transcriptionProvider: selectedTranscriptionProvider,
                    transcriptionModel: normalizedValue,
                });
            }
        },
        [persistState, selectedTranscriptionProvider],
    );

    const clearCachedState = useCallback(() => {
        cachedStateRef.current = null;
        setCurrentBatchJobId(null);
        hasResumedRef.current = false;
        setSelectedFileNames([]);
        const defaultProviderEntry = DEFAULT_TRANSCRIPTION_PROVIDERS[0];
        const providerId = defaultProviderEntry.provider_id;
        const models = defaultProviderEntry.models || [];
        const defaultModel = defaultProviderEntry.default_model || models[0]?.id || FALLBACK_TRANSCRIPTION_MODEL;
        setTranscriptionProviders(DEFAULT_TRANSCRIPTION_PROVIDERS);
        setSelectedTranscriptionProvider(providerId);
        setTranscriptionModels(models);
        setSelectedTranscriptionModel(defaultModel);
        selectedTranscriptionProviderRef.current = providerId;
        selectedTranscriptionModelRef.current = defaultModel;
        if (typeof window !== 'undefined' && window.sessionStorage) {
            try {
                window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
            } catch (storageError) {
                console.warn('SessionStorage clear failed:', storageError);
            }
        }
    }, []);

    useEffect(() => {
        let isSubscribed = true;

        const applyDefaultProviders = () => {
            const defaultProviders = DEFAULT_TRANSCRIPTION_PROVIDERS;
            setTranscriptionProviders(defaultProviders);
            const providerEntry =
                defaultProviders.find(
                    (provider) => provider.provider_id === selectedTranscriptionProviderRef.current,
                ) || defaultProviders[0];
            const providerId = providerEntry.provider_id;
            const models = providerEntry.models || [];
            setSelectedTranscriptionProvider(providerId);
            setTranscriptionModels(models);
            const nextModel =
                normalizeModelValue(providerId, selectedTranscriptionModelRef.current) ||
                providerEntry.default_model ||
                models[0]?.id ||
                FALLBACK_TRANSCRIPTION_MODEL;
            setSelectedTranscriptionModel(nextModel);
            persistState({
                transcriptionProvider: providerId,
                transcriptionModel: nextModel,
            });
            setIsLoadingTranscriptionOptions(false);
        };

        if (!isAuthenticated) {
            applyDefaultProviders();
            return () => {
                isSubscribed = false;
            };
        }

        const fetchTranscriptionProviders = async () => {
            setIsLoadingTranscriptionOptions(true);
            try {
                const response = await authorizedFetch('/transcription/providers/');
                if (!response || !response.ok) {
                    throw new Error(`Status ${response?.status ?? 'unknown'}`);
                }
                const rawBody = await response.text();
                let data = null;
                try {
                    data = rawBody ? JSON.parse(rawBody) : null;
                } catch (jsonError) {
                    console.warn('Nie udało się zdekodować JSON dostawców transkrypcji, fallback do domyślnych:', jsonError);
                    data = null;
                }
                if (!isSubscribed) {
                    return;
                }

                const normalizedProviders = Array.isArray(data)
                    ? data
                          .map((provider) => {
                              if (!provider || typeof provider.provider_id !== 'string') {
                                  return null;
                              }
                              const providerId = provider.provider_id.trim().toLowerCase();
                              if (!providerId) {
                                  return null;
                              }
                              const models = Array.isArray(provider.models)
                                  ? provider.models
                                        .map((model) => {
                                            if (!model || !model.id) {
                                                return null;
                                            }
                                            const modelId = String(model.id).trim();
                                            if (!modelId) {
                                                return null;
                                            }
                                            return {
                                                id: modelId,
                                                label: model.label || modelId,
                                            };
                                        })
                                        .filter(Boolean)
                                  : [];
                              return {
                                  provider_id: providerId,
                                  label: provider.label || providerId,
                                  default_model: provider.default_model,
                                  models,
                              };
                          })
                          .filter(Boolean)
                    : [];

                if (!normalizedProviders.length) {
                    applyDefaultProviders();
                    return;
                }

                setTranscriptionProviders(normalizedProviders);

                const providerIds = normalizedProviders.map((provider) => provider.provider_id);
                let nextProvider = selectedTranscriptionProviderRef.current;
                if (!providerIds.includes(nextProvider)) {
                    nextProvider = normalizedProviders[0].provider_id;
                }
                setSelectedTranscriptionProvider(nextProvider);

                const providerEntry =
                    normalizedProviders.find((provider) => provider.provider_id === nextProvider) ||
                    normalizedProviders[0];
                const providerModels = providerEntry?.models || [];
                setTranscriptionModels(providerModels);

                let nextModel = normalizeModelValue(nextProvider, selectedTranscriptionModelRef.current);
                if (!providerModels.some((model) => model.id === nextModel)) {
                    nextModel =
                        normalizeModelValue(nextProvider, providerEntry?.default_model) ||
                        providerModels[0]?.id ||
                        FALLBACK_TRANSCRIPTION_MODEL;
                }

                setSelectedTranscriptionModel(nextModel);
                persistState({
                    transcriptionProvider: nextProvider,
                    transcriptionModel: nextModel,
                });
            } catch (error) {
                console.warn('Nie udało się pobrać listy dostawców transkrypcji:', error);
                if (isSubscribed) {
                    applyDefaultProviders();
                }
            } finally {
                if (isSubscribed) {
                    setIsLoadingTranscriptionOptions(false);
                }
            }
        };

        fetchTranscriptionProviders();

        return () => {
            isSubscribed = false;
        };
    }, [authorizedFetch, isAuthenticated, persistState]);

    const fetchAndUpdateFileDetails = useCallback(
        async (fileId, fallbackName = '', fileJobId = null) => {
            if (!fileId) {
                return;
            }

            try {
                const response = await authorizedFetch(`/processed_files/${fileId}/details`);
                if (!response || !response.ok) {
                    throw new Error(`Status ${response?.status}`);
                }

                const details = await response.json();
                const detailedNameRaw =
                    extractFileDisplayName(details) ||
                    details?.original_filename ||
                    '';
                const detailedName = detailedNameRaw ? detailedNameRaw.trim() : '';
                const fallbackTrimmed = fallbackName ? fallbackName.trim() : '';
                const fallbackLooksLikeId =
                    !!fallbackTrimmed &&
                    (
                        (fileJobId && fallbackTrimmed === fileJobId) ||
                        (fileId && fallbackTrimmed === String(fileId))
                    );
                const fallbackIsPlaceholder =
                    !!fallbackTrimmed &&
                    fallbackTrimmed.toLowerCase() === 'nieznany plik';
                const fallbackLooksTemporary = fallbackLooksLikeId || fallbackIsPlaceholder;
                const chosenName = (() => {
                    if (fallbackTrimmed && !fallbackLooksTemporary) {
                        return fallbackTrimmed;
                    }
                    if (detailedName) {
                        return detailedName;
                    }
                    return fallbackTrimmed;
                })();

                if (chosenName) {
                    const newMap = { ...fileNameMapRef.current };
                    let mapChanged = false;
                    if (fileId && newMap[String(fileId)] !== chosenName) {
                        newMap[String(fileId)] = chosenName;
                        mapChanged = true;
                    }
                    if (fileJobId && newMap[fileJobId] !== chosenName) {
                        newMap[fileJobId] = chosenName;
                        mapChanged = true;
                    }
                    if (mapChanged) {
                        setFileNameMapState(newMap);
                        persistState({ fileNameMap: newMap });
                    }

                    if (detailedName && (!fallbackTrimmed || fallbackLooksTemporary)) {
                        setProcessedResults((prevResults) => {
                            const updated = prevResults.map((result) =>
                                result.file_id === fileId
                                    ? { ...result, original_filename: detailedName }
                                    : result
                            );
                            persistState({ processedResults: updated });
                            return updated;
                        });
                    }

                    if (
                        detailedName &&
                        (!fallbackTrimmed || fallbackLooksTemporary) &&
                        (currentFileNameRef.current === fallbackName || currentFileNameRef.current === '')
                    ) {
                        currentFileNameRef.current = detailedName;
                        setCurrentFileName(detailedName);
                        persistState({ currentFileName: detailedName });
                    }
                }
            } catch (error) {
                console.warn('Nie udało się pobrać szczegółów pliku:', error);
            }
        },
        [authorizedFetch, extractFileDisplayName, persistState, setFileNameMapState]
    );

    const resetStateToInitial = useCallback((options = {}) => {
        const { preserveResults = false, preserveStatus = false } = options;

        if (!preserveStatus) {
            setStatusMessage('Oczekiwanie na plik...');
            setStatusType('info');
        }

        setBatchStatusMessage('');
        setBatchProgressPercentage(0);
        setCompletedFiles(0);
        setTotalFiles(0);
        setCurrentFileName('');
        currentFileNameRef.current = '';
        setCurrentFileProgressPercentage(0);
        setCurrentFileStatusMessage('');
        setCurrentFileWhisperProgressPercentage(0);
        setCurrentFileWhisperStatusMessage('');
        setCurrentFileMinutesProgressPercentage(0);
        setCurrentFileMinutesStatusMessage('');
        setCurrentFileMinutesSection('');
        setSendButtonDisabled(false);
        setIsFileSelected(false);
        setIsProcessingActive(false);
        setSelectedFileNames([]);

        if (!preserveResults) {
            setProcessedResults([]);
        }

        hasResumedRef.current = false;

        if (fileInputRef.current) {
            fileInputRef.current.value = '';
        }

        if (resumeRetryTimeoutRef.current) {
            clearTimeout(resumeRetryTimeoutRef.current);
            resumeRetryTimeoutRef.current = null;
        }
        persistState({
            isProcessingActive: false,
            selectedFileNames: [],
            fileNameMap: preserveResults ? { ...fileNameMapRef.current } : {}
        });
    }, [setStatusMessage, setStatusType, persistState]);

    // Mimic updateStatus function from original JS
    const updateStatus = useCallback((message, type = 'info') => {
        setStatusMessage(message);
        setStatusType(type);
        console.log(`Status (${type}): ${message}`);
        if (message !== 'Wysyłanie pliku(ów)...') {
            persistState({ statusMessage: message, statusType: type });
        }
    }, [persistState]);

    // Main upload form submission
    const handleUploadSubmit = async (e) => {
        e.preventDefault();
        const files = fileInputRef.current.files;
        const fileList = Array.from(files).map((file) => file.name);
        setSelectedFileNames(fileList);
        persistState({ selectedFileNames: fileList, statusMessage: 'Wysyłanie pliku(ów)...', statusType: 'info' });
        // Usunięto początkową walidację plików, ponieważ jest teraz w warunku disabled przycisku
        // if (files.length === 0) {
        //     updateStatus('Proszę wybrać plik(i) do przetworzenia.', 'warning');
        //     return;
        // }

        const formData = new FormData();
        for (const file of files) {
            formData.append('files', file);
        }
        formData.append('transcription_provider', selectedTranscriptionProviderRef.current || FALLBACK_TRANSCRIPTION_PROVIDER);
        formData.append('transcription_model', selectedTranscriptionModelRef.current || '');

        // Pobierz JSON promptów z PromptManagementSection za pomocą refa
        const customPromptsJson = promptManagementRef.current ? promptManagementRef.current.generateCustomPromptJson() : null;
        if (customPromptsJson) {
            formData.append('custom_prompt', customPromptsJson);
        }

        formData.append('chunk_duration', chunkDurationRef.current?.value || '240');
        formData.append('chunk_overlap', chunkOverlapRef.current?.value || '0');

        const providerTokensPayload = {};
        const allowedProviderKeys = ['gemini', 'assemblyai', 'openai'];
        allowedProviderKeys.forEach((key) => {
            const raw = sessionTokens?.[key];
            if (typeof raw === 'string') {
                const trimmed = raw.trim();
                if (trimmed.length > 0) {
                    providerTokensPayload[key] = trimmed;
                }
            }
        });
        if (Object.keys(providerTokensPayload).length > 0) {
            formData.append('provider_tokens', JSON.stringify(providerTokensPayload));
        }

        setSendButtonDisabled(true);
        persistState({ sendButtonDisabled: true });
        updateStatus('Wysyłanie pliku(ów)...', 'info');
        setBatchStatusMessage('');
        persistState({
            batchStatusMessage: '',
            batchProgressPercentage: 0,
            completedFiles: 0,
            totalFiles: 0,
        });

        let batchStarted = false;
        try {
            const uploadResponse = await authorizedFetch('/upload_multiple/', { // Użyj authorizedFetch
                method: 'POST',
                body: formData,
            });

            if (uploadResponse.ok) {
                const uploadResult = await uploadResponse.json();
                updateStatus(uploadResult.message, 'success');

                const uploadedFiles = Array.isArray(uploadResult.uploaded_files) ? uploadResult.uploaded_files : [];
                if (uploadedFiles.length > 0) {
                    const newMap = { ...fileNameMapRef.current };
                    let mapChanged = false;
                    uploadedFiles.forEach((fileInfo) => {
                        if (fileInfo && fileInfo.file_job_id && fileInfo.filename) {
                            if (newMap[fileInfo.file_job_id] !== fileInfo.filename) {
                                newMap[fileInfo.file_job_id] = fileInfo.filename;
                                mapChanged = true;
                            }
                        }
                    });
                    if (mapChanged) {
                        setFileNameMapState(newMap);
                        persistState({ fileNameMap: newMap });
                    }
                }

                // Jeśli to pojedynczy plik, przekieruj od razu
                if (uploadResult.file_id) {
                    window.open(`/processed_files/${uploadResult.file_id}/details`, '_blank');
                    setSendButtonDisabled(false);
                    persistState({ sendButtonDisabled: false });
                    return; // Zakończ, ponieważ przekierowujemy
                }

                // Jeśli to przetwarzanie wsadowe, kontynuuj strumień
                const batchJobId = uploadResult.batch_job_id;
                batchStarted = true;
                setCurrentBatchJobId(batchJobId);
                setBatchStatusMessage(`Batch Job ID: ${batchJobId}`);
                persistState({
                    batchJobId,
                    batchStatusMessage: `Batch Job ID: ${batchJobId}`,
                });

                // EventSource (SSE) nie obsługuje nagłówków autoryzacji bezpośrednio
                // Więc to połączenie nie będzie autoryzowane w ten sposób.
                // Wymagałoby to np. przekazania tokena w query params lub użycia WebSocketów.
                await processBatch(batchJobId);

            } else {
                const errorData = await uploadResponse.json();
                updateStatus(`Błąd podczas wysyłania pliku: ${errorData.detail || uploadResponse.statusText}`, 'error');
                setSendButtonDisabled(false);
                persistState({ sendButtonDisabled: false });
            }
        } catch (error) {
            updateStatus(`Wystąpił błąd sieci podczas wysyłania pliku: ${error.message}`, 'error');
            setSendButtonDisabled(false);
            persistState({ sendButtonDisabled: false });
            batchStarted = false;
        } finally {
            if (!batchStarted) {
                setSendButtonDisabled(false);
                persistState({ sendButtonDisabled: false });
            }
        }
    };

    // SSE processing - this will need to be adapted for React state
    const processBatch = useCallback(async (batchJobId, options = {}) => {
        const { resume = false } = options;
        const isResumeAttempt = !!resume;

        setCurrentBatchJobId(batchJobId);
        if (resumeRetryTimeoutRef.current) {
            clearTimeout(resumeRetryTimeoutRef.current);
            resumeRetryTimeoutRef.current = null;
        }
        setIsProcessingActive(true);
        persistState({ batchJobId, sendButtonDisabled: true, isProcessingActive: true });

        if (eventSourceRef.current) {
            try {
                eventSourceRef.current.close();
            } catch (closeError) {
                console.warn('EventSource close failed:', closeError);
            }
            eventSourceRef.current = null;
        }

        if (!resume) {
            updateStatus('Rozpoczynanie przetwarzania wsadowego...', 'info');
            const initialState = {
                batchStatusMessage: 'Status wsadowy: Rozpoczynanie...',
                batchProgressPercentage: 0,
                completedFiles: 0,
                totalFiles: 0,
                currentFileName: '',
                currentFileProgressPercentage: 0,
                currentFileStatusMessage: '',
                currentFileWhisperProgressPercentage: 0,
                currentFileWhisperStatusMessage: '',
                currentFileMinutesProgressPercentage: 0,
                currentFileMinutesStatusMessage: '',
                currentFileMinutesSection: '',
                processedResults: [],
                isProcessingActive: true,
            };
            setBatchStatusMessage(initialState.batchStatusMessage);
            setBatchProgressPercentage(initialState.batchProgressPercentage);
            setCompletedFiles(initialState.completedFiles);
            setTotalFiles(initialState.totalFiles);
            setProcessedResults(initialState.processedResults);
            setCurrentFileName(initialState.currentFileName);
            currentFileNameRef.current = '';
            setCurrentFileProgressPercentage(initialState.currentFileProgressPercentage);
            setCurrentFileStatusMessage(initialState.currentFileStatusMessage);
            setCurrentFileWhisperProgressPercentage(initialState.currentFileWhisperProgressPercentage);
            setCurrentFileWhisperStatusMessage(initialState.currentFileWhisperStatusMessage);
            setCurrentFileMinutesProgressPercentage(initialState.currentFileMinutesProgressPercentage);
            setCurrentFileMinutesStatusMessage(initialState.currentFileMinutesStatusMessage);
            setCurrentFileMinutesSection(initialState.currentFileMinutesSection);
            persistState({ ...initialState, batchJobId, sendButtonDisabled: true });
        }

        setSendButtonDisabled(true);

        let eventSourceClosedIntentionally = false;
        let connectionEstablished = false;

        try {
            const eventSource = new EventSource(`/process_batch_status/stream/${batchJobId}/?token=${token}`);
            eventSourceRef.current = eventSource;
            console.log('EventSource connection opened for batch:', batchJobId);

            eventSource.onopen = function() {
                if (isResumeAttempt) {
                    hasResumedRef.current = true;
                }
                console.log('EventSource: Connection established.');
                connectionEstablished = true;
            };

            eventSource.onmessage = function (event) {
                connectionEstablished = true;
                const data = JSON.parse(event.data);
                console.log('EventSource: Received SSE data:', data);
                const persistPayload = {};
                const fileDisplayName = extractFileDisplayName(data);
                const resolvedFileName =
                    fileDisplayName || (currentFileNameRef.current && currentFileNameRef.current.trim()) || '';
                const displayNameWithFallback = resolvedFileName || 'Nieznany plik';

                if (data.current_file_name) {
                    if (data.current_file_name !== currentFileNameRef.current) {
                        currentFileNameRef.current = data.current_file_name;
                        setCurrentFileName(data.current_file_name);
                        setCurrentFileProgressPercentage(0);
                        setCurrentFileStatusMessage('');
                        setCurrentFileWhisperProgressPercentage(0);
                        setCurrentFileWhisperStatusMessage('');
                        setCurrentFileMinutesProgressPercentage(0);
                        setCurrentFileMinutesStatusMessage('');
                        setCurrentFileMinutesSection('');

                        persistPayload.currentFileName = data.current_file_name;
                        persistPayload.currentFileProgressPercentage = 0;
                        persistPayload.currentFileStatusMessage = '';
                        persistPayload.currentFileWhisperProgressPercentage = 0;
                        persistPayload.currentFileWhisperStatusMessage = '';
                        persistPayload.currentFileMinutesProgressPercentage = 0;
                        persistPayload.currentFileMinutesStatusMessage = '';
                        persistPayload.currentFileMinutesSection = '';
                    } else {
                        setCurrentFileName(data.current_file_name);
                        persistPayload.currentFileName = data.current_file_name;
                    }
                }

                if (!data.current_file_name && fileDisplayName) {
                    if (fileDisplayName !== currentFileNameRef.current) {
                        currentFileNameRef.current = fileDisplayName;
                    }
                    setCurrentFileName(fileDisplayName);
                    persistPayload.currentFileName = fileDisplayName;
                }

                if (data.batch_progress) {
                    const completed = data.batch_progress.completed_files;
                    const total = data.batch_progress.total_files;
                    const batchPercentage = total > 0 ? (completed / total) * 100 : 0;

                    setCompletedFiles(completed);
                    setTotalFiles(total);
                    setBatchProgressPercentage(batchPercentage);

                    persistPayload.completedFiles = completed;
                    persistPayload.totalFiles = total;
                    persistPayload.batchProgressPercentage = batchPercentage;

                    const batchStatusText = `Status wsadowy: ${completed}/${total}${data.message ? ` - ${data.message}` : ''}`;
                    setBatchStatusMessage(batchStatusText);
                    persistPayload.batchStatusMessage = batchStatusText;

                    if (data.message) {
                        setCurrentFileStatusMessage(data.message);
                        persistPayload.currentFileStatusMessage = data.message;
                    }
                }

                if (data.status_type === 'info' && data.message) {
                    setCurrentFileStatusMessage(data.message);
                    persistPayload.currentFileStatusMessage = data.message;

                    if (typeof data.current_file_progress_percentage === 'number') {
                        setCurrentFileProgressPercentage(data.current_file_progress_percentage);
                        persistPayload.currentFileProgressPercentage = data.current_file_progress_percentage;
                    }

                    if (typeof data.current_file_whisper_progress_percentage === 'number') {
                        const whisperPercent = data.current_file_whisper_progress_percentage;
                        setCurrentFileWhisperProgressPercentage(whisperPercent);
                        const whisperStatus = `Transkrypcja: ${whisperPercent.toFixed(0)}%`;
                        setCurrentFileWhisperStatusMessage(whisperStatus);
                        persistPayload.currentFileWhisperProgressPercentage = whisperPercent;
                        persistPayload.currentFileWhisperStatusMessage = whisperStatus;
                    }

                    if (typeof data.current_file_minutes_progress_percentage === 'number') {
                        const minutesPercent = data.current_file_minutes_progress_percentage;
                        setCurrentFileMinutesProgressPercentage(minutesPercent);
                        let minutesStatus = `Generowanie minut: ${minutesPercent.toFixed(0)}%`;
                        if (data.current_file_minutes_section) {
                            minutesStatus += ` (sekcja: ${data.current_file_minutes_section})`;
                        }
                        setCurrentFileMinutesStatusMessage(minutesStatus);
                        persistPayload.currentFileMinutesProgressPercentage = minutesPercent;
                        persistPayload.currentFileMinutesStatusMessage = minutesStatus;
                    }

                    if ('current_file_minutes_section' in data) {
                        const section = data.current_file_minutes_section || '';
                        setCurrentFileMinutesSection(section);
                        persistPayload.currentFileMinutesSection = section;
                    }

                    if (
                        data.current_file_progress_percentage === undefined &&
                        data.current_file_whisper_progress_percentage === undefined &&
                        data.current_file_minutes_progress_percentage === undefined
                    ) {
                        setCurrentFileProgressPercentage(0);
                        persistPayload.currentFileProgressPercentage = 0;
                    }
                } else if (data.status_type === 'success') {
                    if (resolvedFileName) {
                        currentFileNameRef.current = resolvedFileName;
                    }
                    setCurrentFileName(displayNameWithFallback);
                    persistPayload.currentFileName = displayNameWithFallback;
                    setCurrentFileProgressPercentage(100);
                    setCurrentFileWhisperProgressPercentage(100);
                    setCurrentFileMinutesProgressPercentage(100);
                    setCurrentFileWhisperStatusMessage('Transkrypcja: 100%');
                    setCurrentFileMinutesStatusMessage('Generowanie minut: 100%');
                    setCurrentFileMinutesSection('');
                    const successMessage = `Plik ${displayNameWithFallback} przetworzony pomyślnie.`;
                    setCurrentFileStatusMessage(successMessage);

                    persistPayload.currentFileProgressPercentage = 100;
                    persistPayload.currentFileWhisperProgressPercentage = 100;
                    persistPayload.currentFileMinutesProgressPercentage = 100;
                    persistPayload.currentFileWhisperStatusMessage = 'Transkrypcja: 100%';
                    persistPayload.currentFileMinutesStatusMessage = 'Generowanie minut: 100%';
                    persistPayload.currentFileMinutesSection = '';
                    persistPayload.currentFileStatusMessage = successMessage;
                } else if (data.status_type === 'error') {
                    if (resolvedFileName) {
                        currentFileNameRef.current = resolvedFileName;
                    }
                    setCurrentFileName(displayNameWithFallback);
                    persistPayload.currentFileName = displayNameWithFallback;
                    setCurrentFileProgressPercentage(0);
                    setCurrentFileWhisperProgressPercentage(0);
                    setCurrentFileMinutesProgressPercentage(0);
                    setCurrentFileWhisperStatusMessage('');
                    setCurrentFileMinutesStatusMessage('');
                    setCurrentFileMinutesSection('');
                    const errorMessage = `Błąd w pliku ${displayNameWithFallback}: ${data.message}`;
                    setCurrentFileStatusMessage(errorMessage);

                    persistPayload.currentFileProgressPercentage = 0;
                    persistPayload.currentFileWhisperProgressPercentage = 0;
                    persistPayload.currentFileMinutesProgressPercentage = 0;
                    persistPayload.currentFileWhisperStatusMessage = '';
                    persistPayload.currentFileMinutesStatusMessage = '';
                    persistPayload.currentFileMinutesSection = '';
                    persistPayload.currentFileStatusMessage = errorMessage;
                }

                if (data.status_type === 'info') {
                    updateStatus(data.message, 'info');
                    if (data.batch_progress) {
                        const infoStatus = `Status wsadowy: ${data.batch_progress.completed_files}/${data.batch_progress.total_files} - ${data.message}`;
                        setBatchStatusMessage(infoStatus);
                        persistPayload.batchStatusMessage = infoStatus;
                    }
                } else if (data.status_type === 'success') {
                    updateStatus(`Plik ${displayNameWithFallback} przetworzony pomyślnie.`, 'success');
                    if (data.batch_progress) {
                        const successStatus = `Status wsadowy: ${data.batch_progress.completed_files}/${data.batch_progress.total_files} - ${data.message}`;
                        setBatchStatusMessage(successStatus);
                        persistPayload.batchStatusMessage = successStatus;
                    }
                    setProcessedResults(prevResults => {
                        const updatedResults = [
                            ...prevResults,
                            {
                                file_id: data.file_id,
                                file_job_id: data.file_job_id ?? null,
                                original_filename: displayNameWithFallback,
                                transcription_filename: data.transcription_filename,
                                minutes_filename: data.minutes_filename,
                                status_type: 'success'
                            }
                        ];
                        persistState({ processedResults: updatedResults });
                        return updatedResults;
                    });
                    const updatedMap = { ...fileNameMapRef.current };
                    let mapChanged = false;
                    if (data.file_job_id && updatedMap[data.file_job_id] !== displayNameWithFallback) {
                        updatedMap[data.file_job_id] = displayNameWithFallback;
                        mapChanged = true;
                    }
                    if (data.file_id && updatedMap[String(data.file_id)] !== displayNameWithFallback) {
                        updatedMap[String(data.file_id)] = displayNameWithFallback;
                        mapChanged = true;
                    }
                    if (mapChanged) {
                        setFileNameMapState(updatedMap);
                        persistState({ fileNameMap: updatedMap });
                    }
                    if (data.file_id) {
                        const trimmedDisplay = displayNameWithFallback ? displayNameWithFallback.trim() : '';
                        const looksLikeJobId =
                            !!trimmedDisplay && !!data.file_job_id && trimmedDisplay === data.file_job_id;
                        const isUnknownPlaceholder =
                            !!trimmedDisplay &&
                            trimmedDisplay.toLowerCase() === 'nieznany plik';
                        if (!trimmedDisplay || looksLikeJobId || isUnknownPlaceholder) {
                            void fetchAndUpdateFileDetails(data.file_id, displayNameWithFallback, data.file_job_id ?? null);
                        }
                    }
                } else if (data.status_type === 'error') {
                    updateStatus(`Błąd podczas przetwarzania pliku ${displayNameWithFallback}: ${data.message}`, 'error');
                    if (data.batch_progress) {
                        const errorStatus = `Status wsadowy: ${data.batch_progress.completed_files}/${data.batch_progress.total_files} - Błąd przetwarzania pliku ${displayNameWithFallback}.`;
                        setBatchStatusMessage(errorStatus);
                        persistPayload.batchStatusMessage = errorStatus;
                    }
                    setProcessedResults(prevResults => {
                        const updatedResults = [
                            ...prevResults,
                            {
                                file_id: data.file_id || 'unknown',
                                file_job_id: data.file_job_id ?? null,
                                original_filename: displayNameWithFallback,
                                status_type: 'error',
                                message: data.message
                            }
                        ];
                        persistState({ processedResults: updatedResults });
                        return updatedResults;
                    });
                    if (data.file_job_id || data.file_id) {
                        const updatedMap = { ...fileNameMapRef.current };
                        let mapChanged = false;
                        if (data.file_job_id && updatedMap[data.file_job_id] !== displayNameWithFallback) {
                            updatedMap[data.file_job_id] = displayNameWithFallback;
                            mapChanged = true;
                        }
                        if (data.file_id && updatedMap[String(data.file_id)] !== displayNameWithFallback) {
                            updatedMap[String(data.file_id)] = displayNameWithFallback;
                            mapChanged = true;
                        }
                        if (mapChanged) {
                            setFileNameMapState(updatedMap);
                            persistState({ fileNameMap: updatedMap });
                        }
                    }
                }

                if (data.batch_complete) {
                    updateStatus('Przetwarzanie wsadowe zakończone!', 'success');
                    setBatchStatusMessage('Status wsadowy: Zakończono!');
                    setCurrentFileProgressPercentage(100);
                    setCurrentFileWhisperProgressPercentage(100);
                    setCurrentFileMinutesProgressPercentage(100);
                    setCurrentFileWhisperStatusMessage('Transkrypcja: 100%');
                    setCurrentFileMinutesStatusMessage('Generowanie minut: 100%');
                    setCurrentFileStatusMessage('Wszystkie pliki przetworzone.');
                    setCurrentFileMinutesSection('');
                    setSendButtonDisabled(false);

                    persistState({
                        batchStatusMessage: 'Status wsadowy: Zakończono!',
                        currentFileProgressPercentage: 100,
                        currentFileWhisperProgressPercentage: 100,
                        currentFileMinutesProgressPercentage: 100,
                        currentFileWhisperStatusMessage: 'Transkrypcja: 100%',
                        currentFileMinutesStatusMessage: 'Generowanie minut: 100%',
                        currentFileStatusMessage: 'Wszystkie pliki przetworzone.',
                        currentFileMinutesSection: '',
                        sendButtonDisabled: false,
                    });

                    eventSourceClosedIntentionally = true;
                    eventSource.close();
                    eventSourceRef.current = null;
                    console.log('EventSource: SSE connection closed due to batch_complete (intentionally).');
                    clearCachedState();
                    resetStateToInitial({ preserveResults: true, preserveStatus: true });

                    if (data.status_type === 'success') {
                        window.open(`/processed_batches/${batchJobId}/details`, '_blank');
                    }
                    return;
                }

                if (Object.keys(persistPayload).length > 0) {
                    persistState(persistPayload);
                }
            };

            eventSource.onerror = function (err) {
                console.error('EventSource: Error occurred. eventSourceClosedIntentionally:', eventSourceClosedIntentionally, 'Error:', err);
                let shouldPersistSendButtonState = true;
                if (eventSourceClosedIntentionally) {
                    console.log('EventSource: SSE connection closed intentionally as batch completed (error suppressed).');
                } else {
                    console.error('EventSource: SSE connection closed due to unexpected error. This might be a network issue or server error.');
                    if (isResumeAttempt && !connectionEstablished) {
                        console.warn('EventSource resume failed before establishing connection. Retrying...');
                        hasResumedRef.current = false;
                        setIsProcessingActive(true);
                        persistState({ isProcessingActive: true });
                        if (resumeRetryTimeoutRef.current) {
                            clearTimeout(resumeRetryTimeoutRef.current);
                        }
                        resumeRetryTimeoutRef.current = setTimeout(() => {
                            if (!eventSourceRef.current) {
                                processBatch(batchJobId, { resume: true });
                            }
                        }, 1500);
                        shouldPersistSendButtonState = false;
                    } else {
                        updateStatus('Połączenie ze strumieniem statusu utracone lub błąd. Sprawdź konsolę po więcej szczegółów.', 'error');
                    }
                }
                if (eventSource) eventSource.close();
                if (eventSourceRef.current === eventSource) {
                    eventSourceRef.current = null;
                }
                setSendButtonDisabled(false);
                if (shouldPersistSendButtonState) {
                    persistState({ sendButtonDisabled: false });
                }
            };
        } catch (error) {
            console.error('EventSource: Error setting up EventSource:', error);
            updateStatus(`Wystąpił błąd podczas konfiguracji strumienia statusu: ${error.message}`, 'error');
            setSendButtonDisabled(false);
            persistState({ sendButtonDisabled: false });
            if (eventSourceRef.current) {
                eventSourceRef.current.close();
                eventSourceRef.current = null;
            }
        }
    }, [updateStatus, token, persistState, clearCachedState, resetStateToInitial, extractFileDisplayName, fetchAndUpdateFileDetails]);

    useEffect(() => {
        if (!initialCachedState) {
            return;
        }

        let cancelled = false;

        const resumeIfNeeded = async () => {
            const cachedJobId = initialCachedState.batchJobId;

            if (!cachedJobId) {
                if (!cancelled) {
                    resetStateToInitial();
                    clearCachedState();
                }
                return;
            }

            if (hasResumedRef.current) {
                return;
            }

            try {
                const detailsResponse = await authorizedFetch(`/processed_batches/${cachedJobId}/details`);
                if (!cancelled && detailsResponse && detailsResponse.ok) {
                    const batchDetails = await detailsResponse.json();
                    const batchStatus = batchDetails?.status;
                    if (batchStatus === 'completed' || batchStatus === 'error') {
                        updateStatus('Przetwarzanie wsadowe zakończone!', 'success');
                        resetStateToInitial({ preserveResults: true, preserveStatus: true });
                        clearCachedState();
                        return;
                    }
                    if (batchStatus === 'processing') {
                        hasResumedRef.current = true;
                        updateStatus('Wznawianie monitorowania postępu...', 'info');
                        setCurrentBatchJobId(cachedJobId);
                        setIsProcessingActive(true);
                        persistState({ isProcessingActive: true, batchJobId: cachedJobId });
                        if (resumeRetryTimeoutRef.current) {
                            clearTimeout(resumeRetryTimeoutRef.current);
                            resumeRetryTimeoutRef.current = null;
                        }
                        resumeRetryTimeoutRef.current = setTimeout(() => {
                            processBatch(cachedJobId, { resume: true });
                        }, 200);
                        return;
                    }
                }
            } catch (detailsError) {
                console.warn('Batch resume details check failed:', detailsError);
            }

            if (cancelled) {
                return;
            }

            hasResumedRef.current = true;
            updateStatus('Wznawianie monitorowania postępu...', 'info');
            setCurrentBatchJobId(cachedJobId);
            setIsProcessingActive(true);
            persistState({ isProcessingActive: true, batchJobId: cachedJobId });
            processBatch(cachedJobId, { resume: true });
        };

        resumeIfNeeded();

        return () => {
            cancelled = true;
            if (resumeRetryTimeoutRef.current) {
                clearTimeout(resumeRetryTimeoutRef.current);
                resumeRetryTimeoutRef.current = null;
            }
        };
    }, [initialCachedState, authorizedFetch, resetStateToInitial, clearCachedState, processBatch, updateStatus, persistState]);

    useEffect(() => {
        return () => {
            if (eventSourceRef.current) {
                try {
                    eventSourceRef.current.close();
                } catch (closeError) {
                    console.warn('EventSource close failed on unmount:', closeError);
                }
                eventSourceRef.current = null;
            }
            if (resumeRetryTimeoutRef.current) {
                clearTimeout(resumeRetryTimeoutRef.current);
                resumeRetryTimeoutRef.current = null;
            }
        };
    }, []);


    const shouldShowProgressSection = (
        totalFiles > 0 ||
        batchProgressPercentage > 0 ||
        completedFiles > 0 ||
        currentFileName ||
        currentFileStatusMessage ||
        currentFileWhisperStatusMessage ||
        currentFileMinutesStatusMessage ||
        currentFileMinutesSection
    );

    const hasAnyProgress =
        isProcessingActive ||
        shouldShowProgressSection ||
        processedResults.length > 0 ||
        batchStatusMessage;

    const statusClasses = STATUS_STYLE_MAP[statusType] ?? STATUS_STYLE_MAP.default;

    return (
        <PageLayout
            title="Audio/Video Processing"
            description="Wgrywaj pliki, monitoruj postęp i pobieraj wyniki przetwarzania."
        >
            <div
                className={`rounded-2xl border border-gray-200 bg-white shadow-md transition-all ${
                    hasAnyProgress ? 'min-h-[calc(100vh-12rem)]' : ''
                }`}
            >
                <div className="flex flex-col gap-6 p-6">

                        <div className="flex flex-wrap gap-2 rounded-xl bg-gray-100 p-1">
                            <button
                                type="button"
                                className={`flex-1 min-w-[180px] rounded-lg px-4 py-2 text-sm font-semibold transition ${
                                    activeTab === 'mainProcessing'
                                        ? 'bg-white text-gray-900 shadow-sm'
                                        : 'text-gray-600 hover:text-gray-800'
                                }`}
                                onClick={() => setActiveTab('mainProcessing')}
                            >
                                Główne Przetwarzanie
                            </button>
                            <button
                                type="button"
                                className={`flex-1 min-w-[180px] rounded-lg px-4 py-2 text-sm font-semibold transition ${
                                    activeTab === 'recognizer'
                                        ? 'bg-white text-gray-900 shadow-sm'
                                        : 'text-gray-600 hover:text-gray-800'
                                }`}
                                onClick={() => setActiveTab('recognizer')}
                            >
                                Rozpoznawanie Mówców
                            </button>
                        </div>

                        <div
                            id="mainProcessing"
                            className={activeTab === 'mainProcessing' ? 'flex flex-col gap-6' : 'hidden'}
                        >
                            {!isAuthenticated && (
                                <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
                                    Musisz być zalogowany, aby korzystać z tej funkcji.&nbsp;
                                    <Link to="/login" className="font-semibold underline">
                                        Zaloguj się
                                    </Link>
                                </div>
                            )}

                            <form
                                id="uploadForm"
                                onSubmit={handleUploadSubmit}
                                className={`space-y-6 ${!isAuthenticated ? 'pointer-events-none opacity-50' : ''}`}
                            >
                                <section className="space-y-4 rounded-xl border border-gray-200 bg-gray-50 p-5 shadow-sm">
                                    <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
                                        <div className="space-y-1">
                                            <label htmlFor="fileInput" className="text-sm font-medium text-gray-700">
                                                Wybierz plik(i) audio/wideo
                                            </label>
                                            <p className="text-xs text-gray-500">
                                                Obsługiwane formaty: WAV, MP3, FLAC, AAC, OGG, M4A, MKV, MP4, AVI, MOV,
                                                FLV, WMV.
                                            </p>
                                        </div>
                                        <div className="flex items-center gap-3">
                                            <input
                                                type="file"
                                                id="fileInput"
                                                accept=".wav,.mp3,.flac,.aac,.ogg,.m4a,.mkv,.mp4,.avi,.mov,.flv,.wmv"
                                                required
                                                multiple
                                                ref={fileInputRef}
                                                className="hidden"
                                                onChange={(e) => {
                                                    if (e.target.files.length > 0) {
                                                        const fileNames = Array.from(e.target.files).map(
                                                            (file) => file.name
                                                        );
                                                        updateStatus(
                                                            `${e.target.files.length} plik(i) wybranych.`,
                                                            'info'
                                                        );
                                                        setIsFileSelected(true);
                                                        setSelectedFileNames(fileNames);
                                                        persistState({ selectedFileNames: fileNames });
                                                    } else {
                                                        updateStatus('Oczekiwanie na plik...', 'info');
                                                        setIsFileSelected(false);
                                                        setSelectedFileNames([]);
                                                        persistState({ selectedFileNames: [] });
                                                    }
                                                }}
                                            />
                                            <label
                                                htmlFor="fileInput"
                                                className="inline-flex items-center rounded-md bg-gray-900 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-gray-800"
                                            >
                                                Przeglądaj pliki
                                            </label>
                                        </div>
                                    </div>
                                    {selectedFileNames.length > 0 && (
                                        <div className="flex flex-wrap items-center gap-2">
                                            <span className="text-xs font-semibold uppercase tracking-wide text-gray-500">
                                                Wybrane pliki:
                                            </span>
                                            {selectedFileNames.map((name, idx) => (
                                                <span
                                                    key={`${name}-${idx}`}
                                                    className="rounded-full bg-gray-200 px-3 py-1 text-xs font-medium text-gray-900"
                                                >
                                                    {name}
                                                </span>
                                            ))}
                                        </div>
                                    )}

                                    <div className="grid gap-4 sm:grid-cols-2">
                                        <div>
                                            <label
                                                htmlFor="transcriptionProviderSelect"
                                                className="mb-1 block text-sm font-medium text-gray-700"
                                            >
                                                Dostawca transkrypcji
                                            </label>
                                            <select
                                                id="transcriptionProviderSelect"
                                                ref={providerSelectRef}
                                                value={selectedTranscriptionProvider}
                                                onChange={handleTranscriptionProviderChange}
                                                disabled={isLoadingTranscriptionOptions}
                                                className="block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-900/30 disabled:cursor-not-allowed disabled:bg-gray-100"
                                            >
                                                {transcriptionProviders.map((provider) => (
                                                    <option key={provider.provider_id} value={provider.provider_id}>
                                                        {provider.label}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                        <div>
                                            <label
                                                htmlFor="transcriptionModelSelect"
                                                className="mb-1 block text-sm font-medium text-gray-700"
                                            >
                                                Model transkrypcji
                                            </label>
                                            <select
                                                id="transcriptionModelSelect"
                                                ref={transcriptionModelSelectRef}
                                                value={selectedTranscriptionModel}
                                                onChange={handleTranscriptionModelChange}
                                                disabled={isLoadingTranscriptionOptions || transcriptionModels.length === 0}
                                                className="block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-900/30 disabled:cursor-not-allowed disabled:bg-gray-100"
                                            >
                                                {transcriptionModels.map((model) => (
                                                    <option key={model.id} value={model.id}>
                                                        {model.label}
                                                    </option>
                                                ))}
                                            </select>
                                        </div>
                                    </div>
                                    {isLoadingTranscriptionOptions && (
                                        <p className="mt-1 text-xs text-gray-400">Ładowanie listy modeli...</p>
                                    )}
                                </section>

                                <section className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
                                    <button
                                        type="button"
                                        onClick={() => setAdvancedOptionsVisible(!advancedOptionsVisible)}
                                        className="flex w-full items-center justify-between rounded-lg border border-gray-200 bg-gray-50 px-4 py-3 text-left text-sm font-semibold text-gray-700 transition hover:bg-gray-100"
                                    >
                                        <span>Zaawansowane opcje</span>
                                        <span
                                            className={`transform text-lg transition ${
                                                advancedOptionsVisible ? 'rotate-180' : ''
                                            }`}
                                        >
                                            ▼
                                        </span>
                                    </button>

                                    {advancedOptionsVisible && (
                                        <div className="mt-4 space-y-4 border-t border-gray-200 pt-4">
                                            <div className="rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
                                                <PromptManagementSection
                                                    ref={promptManagementRef}
                                                    isAuthenticated={isAuthenticated}
                                                    updateStatus={updateStatus}
                                                />
                                                <input
                                                    type="hidden"
                                                    id="generatedCustomPrompt"
                                                    name="generated_custom_prompt"
                                                    value={
                                                        (promptManagementRef.current &&
                                                            promptManagementRef.current.generateCustomPromptJson()) ||
                                                        ''
                                                    }
                                                />
                                            </div>

                                            <div className="grid gap-4 sm:grid-cols-2">
                                                <div className="space-y-1">
                                                    <label
                                                        htmlFor="chunkDuration"
                                                        className="block text-sm font-medium text-gray-700"
                                                    >
                                                        Długość chunka (sekundy)
                                                    </label>
                                                    <input
                                                        type="number"
                                                        id="chunkDuration"
                                                        defaultValue="240"
                                                        min="60"
                                                        max="600"
                                                        step="30"
                                                        className="block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-900/30"
                                                        ref={chunkDurationRef}
                                                    />
                                                </div>
                                                <div className="space-y-1">
                                                    <label
                                                        htmlFor="chunkOverlap"
                                                        className="block text-sm font-medium text-gray-700"
                                                    >
                                                        Nakładanie chunków (sekundy)
                                                    </label>
                                                    <input
                                                        type="number"
                                                        id="chunkOverlap"
                                                        defaultValue="0"
                                                        min="0"
                                                        max="120"
                                                        step="10"
                                                        className="block w-full rounded-md border border-gray-300 px-3 py-2 text-sm shadow-sm focus:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-900/30"
                                                        ref={chunkOverlapRef}
                                                    />
                                                </div>
                                            </div>
                                        </div>
                                    )}
                                </section>

                                <div className="flex justify-end">
                                    <button
                                        type="submit"
                                        id="submitButton"
                                        className="inline-flex items-center rounded-md border border-gray-700 bg-white px-5 py-2.5 text-sm font-semibold text-gray-900 shadow-md transition hover:bg-gray-100 hover:border-gray-900 focus:outline-none focus:ring-2 focus:ring-gray-400 disabled:cursor-not-allowed disabled:opacity-50"
                                        disabled={sendButtonDisabled || !isAuthenticated || !isFileSelected}
                                    >
                                        Rozpocznij przetwarzanie
                                    </button>
                                </div>
                            </form>
                        </div>

                        <div
                            id="recognizer"
                            className={activeTab === 'recognizer' ? 'block' : 'hidden'}
                        >
                            <SpeakerManagementSection updateStatus={updateStatus} />
                        </div>

                        <div className={`${statusClasses} rounded-xl px-5 py-4 text-sm font-medium`}>
                            {statusMessage}
                        </div>

                        {batchStatusMessage && (
                            <p className="text-sm italic text-gray-500">{batchStatusMessage}</p>
                        )}

                        {shouldShowProgressSection && (
                            <section className="space-y-5 rounded-xl border border-gray-200 bg-gray-50 p-5 shadow-sm">
                                <div className="space-y-2">
                                    <div className="flex items-center justify-between text-sm text-gray-600">
                                        <span>Postęp przetwarzania wsadu</span>
                                        <span className="font-semibold text-gray-800">
                                            {batchProgressPercentage.toFixed(2)}%
                                        </span>
                                    </div>
                                    <div className="h-3 w-full overflow-hidden rounded-full bg-gray-200">
                                        <div
                                            className="h-full rounded-full bg-gray-900 transition-all"
                                            style={{ width: `${batchProgressPercentage}%` }}
                                        ></div>
                                    </div>
                                    <p className="text-xs text-gray-500">
                                        {totalFiles > 0
                                            ? `${completedFiles} / ${totalFiles} plików przetworzonych (${batchProgressPercentage.toFixed(2)}%)`
                                            : `Postęp wsadu: ${batchProgressPercentage.toFixed(2)}%`}
                                    </p>
                                </div>

                                {(currentFileName ||
                                    currentFileStatusMessage ||
                                    currentFileWhisperStatusMessage ||
                                    currentFileMinutesStatusMessage) && (
                                    <div className="space-y-4 rounded-lg border border-gray-200 bg-white p-4 shadow-sm">
                                        {currentFileName && (
                                            <p className="text-sm font-semibold text-gray-800">
                                                Aktualny plik:{' '}
                                                <span className="font-normal text-gray-600">{currentFileName}</span>
                                            </p>
                                        )}

                                        {currentFileStatusMessage && (
                                            <p className="text-xs text-gray-600">{currentFileStatusMessage}</p>
                                        )}

                                        <div className="space-y-2">
                                            <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
                                                <div
                                                    className="h-full rounded-full bg-purple-600 transition-all"
                                                    style={{ width: `${currentFileProgressPercentage}%` }}
                                                ></div>
                                            </div>
                                            <p className="text-xs text-gray-500">
                                                Postęp pliku: {currentFileProgressPercentage.toFixed(0)}%
                                            </p>
                                        </div>

                                        {currentFileWhisperStatusMessage && (
                                            <div className="space-y-2">
                                                <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">
                                                    {currentFileWhisperStatusMessage}
                                                </p>
                                                <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
                                                    <div
                                                        className="h-full rounded-full bg-indigo-500 transition-all"
                                                        style={{ width: `${currentFileWhisperProgressPercentage}%` }}
                                                    ></div>
                                                </div>
                                                <p className="text-xs text-gray-500">
                                                    Postęp Whisper: {currentFileWhisperProgressPercentage.toFixed(0)}%
                                                </p>
                                            </div>
                                        )}

                                        {currentFileMinutesStatusMessage && (
                                            <div className="space-y-2">
                                                <p className="text-xs font-semibold uppercase tracking-wide text-gray-500">
                                                    {currentFileMinutesStatusMessage}
                                                </p>
                                                {currentFileMinutesSection && (
                                                    <p className="text-xs text-gray-400">
                                                        Sekcja: {currentFileMinutesSection}
                                                    </p>
                                                )}
                                                <div className="h-2 w-full overflow-hidden rounded-full bg-gray-200">
                                                    <div
                                                        className="h-full rounded-full bg-green-500 transition-all"
                                                        style={{ width: `${currentFileMinutesProgressPercentage}%` }}
                                                    ></div>
                                                </div>
                                                <p className="text-xs text-gray-500">
                                                    Postęp minut: {currentFileMinutesProgressPercentage.toFixed(0)}%
                                                </p>
                                            </div>
                                        )}
                                    </div>
                                )}
                            </section>
                        )}

                        {processedResults.length > 0 && (
                            <section className="space-y-4 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
                                <h2 className="text-lg font-semibold text-gray-900">Wyniki przetwarzania</h2>
                                {processedResults.map((result, index) => {
                                    const resultDisplayName = extractFileDisplayName(result) || 'Nieznany plik';
                                    return (
                                        <div
                                            key={index}
                                            className="space-y-3 rounded-lg border border-gray-200 bg-gray-50 p-4 shadow-sm"
                                        >
                                            <h4 className="text-base font-semibold text-gray-900">
                                                Plik: {resultDisplayName}
                                            </h4>
                                            {result.status_type === 'success' ? (
                                                <>
                                                    {result.file_id && (
                                                        <p className="text-sm text-gray-700">
                                                            <strong>Szczegóły Joba:</strong>{' '}
                                                            <a
                                                                href={`/processed_files/${result.file_id}/details`}
                                                                target="_blank"
                                                                rel="noopener noreferrer"
                                                                className="text-gray-900 underline hover:text-gray-700"
                                                            >
                                                                Zobacz szczegóły
                                                            </a>
                                                        </p>
                                                    )}
                                                    {result.transcription_filename && (
                                                        <p className="text-sm text-gray-700">
                                                            <strong>Transkrypcja:</strong>{' '}
                                                            <a
                                                                href={`/${result.transcription_filename}`}
                                                                target="_blank"
                                                                rel="noopener noreferrer"
                                                                className="text-gray-900 underline hover:text-gray-700"
                                                            >
                                                                Pobierz plik
                                                            </a>
                                                        </p>
                                                    )}
                                                    {result.minutes_filename && (
                                                        <p className="text-sm text-gray-700">
                                                            <strong>Protokoły:</strong>{' '}
                                                            <a
                                                                href={`/processed_files/${result.file_id}/minutes?format=html`}
                                                                target="_blank"
                                                                rel="noopener noreferrer"
                                                                className="text-gray-900 underline hover:text-gray-700"
                                                            >
                                                                Zobacz HTML
                                                            </a>
                                                            <span className="text-gray-400"> • </span>
                                                            <a
                                                                href={`/${result.minutes_filename}`}
                                                                target="_blank"
                                                                rel="noopener noreferrer"
                                                                className="text-gray-900 underline hover:text-gray-700"
                                                            >
                                                                Pobierz plik
                                                            </a>
                                                        </p>
                                                    )}
                                                </>
                                            ) : (
                                                <p className="text-sm text-red-600">
                                                    Status: Błąd — {result.message}
                                                </p>
                                            )}
                                        </div>
                                    );
                                })}
                            </section>
                        )}
                    </div>
                </div>
        </PageLayout>
    );
}

export default UploadPage;

