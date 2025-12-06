import React, { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { useAuth } from './AuthContext';
import PageLayout from './PageLayout';

function LoginPage() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState(null);
    const navigate = useNavigate();
    const { login } = useAuth();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null);

        try {
            const response = await fetch('/token', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: new URLSearchParams({
                    username,
                    password,
                }).toString(),
            });

            console.log('[LOGIN] Response status:', response.status);
            const data = await response.json();
            console.log('[LOGIN] Response data:', data);

            if (response.ok) {
                login(data.access_token);
                navigate('/');
            } else {
                setError(data.detail || 'Błąd logowania');
            }
        } catch (err) {
            console.error('[LOGIN] Network error:', err);
            setError('Błąd sieci lub serwera');
        }
    };

    return (
        <PageLayout title="Logowanie" description="Uzyskaj dostęp do panelu Sekretarza.">
            <div className="mx-auto w-full max-w-md rounded-xl border border-gray-200 bg-white p-8 shadow-md">
                <form onSubmit={handleSubmit}>
                    <div className="mb-4">
                        <label htmlFor="username" className="block text-gray-700 text-sm font-bold mb-2">Nazwa użytkownika:</label>
                        <input
                            type="text"
                            id="username"
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            required
                        />
                    </div>
                    <div className="mb-6">
                        <label htmlFor="password" className="block text-gray-700 text-sm font-bold mb-2">Hasło:</label>
                        <input
                            type="password"
                            id="password"
                            className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 mb-3 leading-tight focus:outline-none focus:shadow-outline"
                            value={password}
                            onChange={(e) => setPassword(e.target.value)}
                            required
                        />
                    </div>
                    {error && <p className="text-red-500 text-xs italic mb-4 text-center">{error}</p>}
                    <div className="flex items-center justify-between">
                        <button
                            type="submit"
                            className="bg-gray-900 hover:bg-gray-800 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline w-full"
                        >
                            Zaloguj
                        </button>
                    </div>
                </form>
                <p className="mt-6 text-center text-xs text-gray-500">
                    Nie masz konta?{' '}
                    <Link
                        to="/register"
                        className="text-gray-900 underline decoration-transparent hover:decoration-current"
                    >
                        Zarejestruj się
                    </Link>
                </p>
            </div>
        </PageLayout>
    );
}

export default LoginPage;
