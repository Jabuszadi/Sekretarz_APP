const STORAGE_KEY = 'sekretarz.apiTokens.v1';

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder();

function normalizeProviderKey(provider) {
  if (!provider) {
    return '';
  }
  const normalized = String(provider).trim().toLowerCase();
  if (normalized === 'assembly') {
    return 'assemblyai';
  }
  return normalized;
}

function normalizeStoredEntries(raw) {
  if (!raw || typeof raw !== 'object') {
    return {};
  }
  const normalized = {};
  Object.entries(raw).forEach(([key, value]) => {
    const normalizedKey = normalizeProviderKey(key);
    if (!normalizedKey) {
      return;
    }
    normalized[normalizedKey] = value;
  });
  return normalized;
}

function bufferToBase64(buffer) {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return window.btoa(binary);
}

function base64ToBuffer(base64) {
  const binary = window.atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    bytes[i] = binary.charCodeAt(i);
  }
  return bytes;
}

function getStorage() {
  if (typeof window === 'undefined' || !window.localStorage) {
    throw new Error('localStorage is not available in this environment.');
  }
  return window.localStorage;
}

function readStoredObject() {
  try {
    const storage = getStorage();
    const raw = storage.getItem(STORAGE_KEY);
    if (!raw) {
      return {};
    }
    const parsed = JSON.parse(raw);
    return normalizeStoredEntries(parsed);
  } catch (error) {
    console.warn('[encryptedStorage] Nie udało się odczytać danych:', error);
    return {};
  }
}

function writeStoredObject(data) {
  const storage = getStorage();
  storage.setItem(STORAGE_KEY, JSON.stringify(normalizeStoredEntries(data)));
}

async function deriveKey(passphrase, salt) {
  const passphraseKey = await window.crypto.subtle.importKey(
    'raw',
    textEncoder.encode(passphrase),
    'PBKDF2',
    false,
    ['deriveKey'],
  );

  return window.crypto.subtle.deriveKey(
    {
      name: 'PBKDF2',
      salt,
      iterations: 100000,
      hash: 'SHA-256',
    },
    passphraseKey,
    {
      name: 'AES-GCM',
      length: 256,
    },
    false,
    ['encrypt', 'decrypt'],
  );
}

export function listStoredProviders() {
  const stored = readStoredObject();
  return Object.entries(stored).reduce((acc, [provider, entry]) => {
    if (!entry || !entry.ciphertext) {
      return acc;
    }
    acc[provider] = {
      updatedAt: entry.updatedAt ?? null,
    };
    return acc;
  }, {});
}

export function deleteStoredToken(provider) {
  const normalizedProvider = normalizeProviderKey(provider);
  if (!normalizedProvider) {
    return;
  }
  const stored = readStoredObject();
  if (stored[normalizedProvider]) {
    delete stored[normalizedProvider];
    writeStoredObject(stored);
  }
}

export async function encryptAndStoreToken(provider, token, passphrase) {
  const normalizedProvider = normalizeProviderKey(provider);
  if (!normalizedProvider) {
    throw new Error('Niepoprawny identyfikator dostawcy.');
  }
  if (!token) {
    throw new Error('Nie podano klucza API.');
  }
  if (!passphrase || passphrase.length < 6) {
    throw new Error('Hasło musi mieć co najmniej 6 znaków.');
  }

  const salt = window.crypto.getRandomValues(new Uint8Array(16));
  const iv = window.crypto.getRandomValues(new Uint8Array(12));
  const key = await deriveKey(passphrase, salt);
  const ciphertext = await window.crypto.subtle.encrypt(
    { name: 'AES-GCM', iv },
    key,
    textEncoder.encode(token),
  );

  const stored = readStoredObject();
  stored[normalizedProvider] = {
    salt: bufferToBase64(salt),
    iv: bufferToBase64(iv),
    ciphertext: bufferToBase64(ciphertext),
    updatedAt: new Date().toISOString(),
  };
  writeStoredObject(stored);
}

export async function decryptStoredToken(provider, passphrase) {
  const normalizedProvider = normalizeProviderKey(provider);
  if (!normalizedProvider) {
    throw new Error('Niepoprawny identyfikator dostawcy.');
  }
  const stored = readStoredObject();
  const entry = stored[normalizedProvider];
  if (!entry) {
    throw new Error('Brak zapisanego klucza dla wybranego dostawcy.');
  }
  if (!passphrase || passphrase.length < 6) {
    throw new Error('Hasło musi mieć co najmniej 6 znaków.');
  }

  const salt = base64ToBuffer(entry.salt);
  const iv = base64ToBuffer(entry.iv);
  const ciphertext = base64ToBuffer(entry.ciphertext);
  const key = await deriveKey(passphrase, salt);

  const plaintext = await window.crypto.subtle.decrypt(
    { name: 'AES-GCM', iv },
    key,
    ciphertext,
  );

  return textDecoder.decode(plaintext);
}

export function clearAllTokens() {
  const storage = getStorage();
  storage.removeItem(STORAGE_KEY);
}

