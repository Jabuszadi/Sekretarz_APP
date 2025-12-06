import React, { createContext, useCallback, useContext, useMemo, useState } from 'react';

const ApiTokensContext = createContext(null);

export function ApiTokensProvider({ children }) {
  const [tokens, setTokens] = useState({});

  const setToken = useCallback((provider, token) => {
    setTokens((prev) => {
      if (!provider) {
        return prev;
      }
      if (!token) {
        const { [provider]: _removed, ...rest } = prev;
        return rest;
      }
      return {
        ...prev,
        [provider]: token,
      };
    });
  }, []);

  const clearTokens = useCallback(() => {
    setTokens({});
  }, []);

  const value = useMemo(
    () => ({
      tokens,
      setToken,
      clearTokens,
    }),
    [tokens, setToken, clearTokens],
  );

  return <ApiTokensContext.Provider value={value}>{children}</ApiTokensContext.Provider>;
}

export function useApiTokens() {
  const context = useContext(ApiTokensContext);
  if (context === null) {
    throw new Error('useApiTokens must be used within an ApiTokensProvider');
  }
  return context;
}

