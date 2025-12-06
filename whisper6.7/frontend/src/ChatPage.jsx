// frontend/src/App.jsx
import React, { useState, useEffect, useRef, useCallback } from 'react';
import './index.css';
import { marked } from 'marked';
import { useAuthorizedFetch } from './AuthContext';
import PageLayout from './PageLayout';

function ChatPage() {
    const [messages, setMessages] = useState([]); // Zmieniono z chatHistory
    const [input, setInput] = useState(''); // Zmieniono z userQuery
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const messagesEndRef = useRef(null);
    const [sendButtonDisabled, setSendButtonDisabled] = useState(false);
    const authorizedFetch = useAuthorizedFetch();

    // Funkcja do przewijania do najnowszej wiadomości
    const scrollToBottom = useCallback(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [messagesEndRef]);

    useEffect(() => {
        scrollToBottom();
    }, [messages, scrollToBottom]);

    const handleSendMessage = async () => {
        if (input.trim() === '') return;
        const userMessage = { text: input, sender: 'user', timestamp: new Date().toLocaleTimeString() };
        setMessages((prevMessages) => [...prevMessages, userMessage]);
        setInput('');
        setLoading(true);
        setSendButtonDisabled(true);

        try {
            const response = await authorizedFetch('/chat/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query: input }), // Usunięto collection_name
            });

            if (!response.ok) {
                let errorMessage = `HTTP error! status: ${response.status}`;
                try {
                    const errorData = await response.json();
                    errorMessage = errorData.detail || errorMessage;
                } catch (parseError) {
                    console.warn("Nie udało się sparsować odpowiedzi błędu jako JSON:", parseError);
                }
                throw new Error(errorMessage);
            }

            const data = await response.json();
            setMessages((prevMessages) => [
                ...prevMessages,
                { text: marked.parse(data.response), sender: 'ai', timestamp: new Date().toLocaleTimeString() },
            ]);
        } catch (error) {
            console.error("Błąd podczas wysyłania wiadomości:", error);
            setMessages((prevMessages) => [
                ...prevMessages,
                { text: `Błąd: ${error.message}`, sender: 'ai', timestamp: new Date().toLocaleTimeString(), isError: true },
            ]);
            setError(error.message);
        } finally {
            setLoading(false);
            setSendButtonDisabled(false);
        }
    };

    return (
        <PageLayout
            title="Agent Czatowy"
            description="Zadawaj pytania dotyczące transkrypcji i protokołów spotkań."
        >
            <div className="bg-white p-5 rounded-lg shadow-md flex-grow flex flex-col max-h-[calc(100vh-10rem)]">

                {/* Chat History: takes remaining available vertical space, allows scrolling if content overflows */}
                {/* `min-h-0` is important to allow flex-grow to shrink the element if needed, preventing overflow */}
                <div id="chatHistory" className="border border-gray-300 p-4 overflow-y-auto mb-5 bg-gray-50 rounded-lg flex flex-col gap-2 flex-grow min-h-100">
                    {messages.map((message, index) => (
                        <div
                            key={index}
                            className={`mb-2 p-2 rounded-xl max-w-[80%] break-words
                                ${message.sender === 'user'
                                    ? 'bg-gray-900 text-white ml-auto rounded-br-sm'
                                    : 'bg-gray-200 text-gray-800 mr-auto rounded-bl-sm'
                                }`}
                        >
                            {message.sender === 'user' ? (
                                message.text
                            ) : (
                                <div dangerouslySetInnerHTML={{ __html: marked.parse(message.text) }} />
                            )}
                        </div>
                    ))}
                    <div ref={messagesEndRef} />
                </div>

                {loading && <div className="text-center p-2 text-gray-700">Ładowanie...</div>}
                {error && (
                    <div className="mt-5 p-2 rounded-md font-bold whitespace-pre-wrap bg-red-100 text-red-700">
                        Błąd: {error}
                    </div>
                )}
                {/* Input form and status remain at the bottom */}
                <div className="bg-white p-4 border-t border-gray-200 flex items-center">
                    <input
                        type="text"
                        className="flex-grow p-2 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-gray-900 focus:border-gray-900 text-gray-800"
                        placeholder="Wpisz swoją wiadomość..."
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        onKeyPress={(e) => {
                            if (e.key === 'Enter') {
                                handleSendMessage();
                            }
                        }}
                        disabled={loading}
                    />
                    <button
                        className={`ml-2 px-4 py-2 rounded-md text-white ${loading ? 'bg-gray-500 cursor-not-allowed' : 'bg-gray-900 hover:bg-gray-800'}`}
                        onClick={handleSendMessage}
                        disabled={sendButtonDisabled || loading}
                    >
                        Wyślij
                    </button>
                </div>
            </div>
        </PageLayout>
    );
}

export default ChatPage;