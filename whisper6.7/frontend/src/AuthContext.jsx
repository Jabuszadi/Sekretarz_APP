import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom'; // Importuj useNavigate

const AuthContext = createContext(null);

export const AuthProvider = ({ children }) => {
    const [token, setToken] = useState(() => {
        const storedToken = localStorage.getItem('authToken');
        console.log('AuthContext Init - Stored Token:', storedToken);
        return storedToken;
    });
    const [isAuthenticated, setIsAuthenticated] = useState(() => {
        const authStatus = !!token;
        console.log('AuthContext Init - Is Authenticated:', authStatus);
        return authStatus;
    });

    // Function to set token and authentication status
    const login = useCallback((newToken) => {
        localStorage.setItem('authToken', newToken);
        setToken(newToken);
        setIsAuthenticated(true);
    }, []);

    // Function to clear token and authentication status
    const logout = useCallback(() => {
        localStorage.removeItem('authToken');
        setToken(null);
        setIsAuthenticated(false);
    }, []);

    // Effect to handle changes to the token state (e.g., if token is manually cleared from localStorage)
    useEffect(() => {
        setIsAuthenticated(!!token);
    }, [token]);

    // Value provided to children components through context
    const authContextValue = {
        token,
        isAuthenticated,
        login,
        logout,
    };

    return (
        <AuthContext.Provider value={authContextValue}>
            {children}
        </AuthContext.Provider>
    );
};

// Custom hook for easy access to auth context
export const useAuth = () => {
    const context = useContext(AuthContext);
    if (context === undefined) {
        throw new Error('useAuth must be used within an AuthProvider');
    }
    return context;
};

// NOWY CUSTOM HOOK: useAuthorizedFetch
// Używa tokena z AuthContext i logiki przekierowania dla 401
export const useAuthorizedFetch = () => {
    const { token, logout, isAuthenticated } = useAuth(); // Użyj logout z useAuth
    const navigate = useNavigate();

    const authorizedFetch = useCallback(async (url, options = {}) => {
        const headers = {
            ...(options.headers || {}),
        };
        console.log('[AUTH FETCH] Current token (truncated):', token ? token.slice(0, 10) + '...' : 'null');
        if (token) {
            headers['Authorization'] = `Bearer ${token}`;
        }
        console.log('[AUTH FETCH] Request', url, 'with headers:', headers);

        try {
            const response = await fetch(url, { ...options, headers, credentials: "include" }); // DODANO credentials: "include"

            // Obsługa odpowiedzi 401: przekieruj na stronę logowania
            if (response.status === 401) {
                // Możesz dodać tutaj logout(), jeśli chcesz automatycznie wylogować
                logout(); // Automatyczne wylogowanie przy 401
                navigate('/login');
                console.warn('[AUTH FETCH] Received 401 from', url);
                throw new Error('Unauthorized: Przekierowanie do logowania.');
            }

            return response;
        } catch (error) {
            console.error('[AUTH FETCH] Błąd:', error);
            throw error; // Przekaż błąd dalej, aby wywołujący komponent mógł go obsłużyć
        }
    }, [token, navigate, logout]); // Dodaj logout do zależności

    return authorizedFetch;
};
