### Changelog - Wersja [Data bieżąca: 2025-11-03]

To wydanie skupia się na integracji interfejsu użytkownika `UploadPage` z aplikacją React, ujednoliceniu routingu, poprawie stabilności strumieniowania statusu przetwarzania plików oraz optymalizacji interfejsu użytkownika.

---

#### Główne Usprawnienia i Integracja

*   **Migracja Interfejsu Uploadu do Reacta:** Cały interfejs użytkownika do przesyłania i przetwarzania plików (pierwotnie zaimplementowany w `index.html` i obsługiwany przez `api_app.py`) został w pełni zmigrowany do nowej, dedykowanej komponentu React: `frontend/src/UploadPage.jsx`. Obejmuje to zarządzanie przesyłaniem plików, niestandardowymi promptami, rejestracją i usuwaniem mówców oraz wyświetlaniem statusu przetwarzania.
*   **Wprowadzenie React Router:** Zaimplementowano `react-router-dom` w `frontend/src/main.jsx` i `frontend/src/App.jsx`, aby umożliwić nawigację między dwiema głównymi częściami aplikacji:
    *   `/chat` dla interfejsu czatu (`ChatPage.jsx`)
    *   `/upload` dla interfejsu przesyłania i przetwarzania plików (`UploadPage.jsx`)
    *   Domyślna ścieżka `/` również kieruje do `ChatPage.jsx`.
*   **Ujednolicenie i Czyszczenie Backendu:** Potwierdzono i uporządkowano odpowiedzialności backendu: `minimal_mcp_server.py` obsługuje wyłącznie logikę czatu, natomiast `api_app.py` jest dedykowany do obsługi uploadu, przetwarzania plików i zarządzania mówcami. Usunięto zbędną logikę czatu z `api_app.py`.

---

#### Frontend (`frontend/`)

**`frontend/src/App.jsx`**
*   **Zmiana:** Usunięto `min-h-screen` z najbardziej zewnętrznego `div`, zmieniono `max-w-6xl` na `max-w-7xl` na elemencie `<main>` i usunięto `p-5` z `<main>`.
*   **Cel:** Umożliwienie pełnej szerokości tła strony przy zachowaniu węższych, wyśrodkowanych okien chatu i uploadu, oraz poprawa zarządzania wysokością i przewijaniem.

**`frontend/src/ChatPage.jsx`**
*   **Zmiana:** Usunięto selektor kolekcji Qdrant (i związaną z nim logikę) z UI czatu.
*   **Cel:** Uproszczenie interfejsu, ponieważ będzie używana tylko jedna kolekcja Qdrant.
*   **Zmiana:** Usunięto `mx-auto` z głównego `div`, zmieniono zewnętrzny `div` z `min-h-screen` na `w-full h-full flex flex-col px-5`. Dodano `max-h-[calc(100vh-10rem)]` do wewnętrznego `div` (`bg-white p-5 rounded-lg shadow-md flex-grow flex flex-col`).
*   **Cel:** Dostosowanie szerokości i wyśrodkowania przez `App.jsx`, zarządzanie wysokością okna czatu, aby nie wychodziło poza widok i prawidłowe przewijanie wewnętrzne.
*   **Zmiana:** Dodano klasę Tailwind CSS `text-gray-800` do elementu `<input>` pola wprowadzania czatu.
*   **Cel:** Naprawienie problemu z niewidocznym (białym) tekstem w polu wprowadzania czatu.

**`frontend/src/UploadPage.jsx`**
*   **Zmiana:** Usunięto `mx-auto` i `max-w-4xl` z głównego `div`. Dodano `px-5` do najbardziej zewnętrznego `div` (`<div className="w-full h-full flex flex-col px-5">`). Dodano `max-h-[calc(100vh-10rem)]` do wewnętrznego `div` (`<div className="bg-white p-5 rounded-lg shadow-md flex-grow flex flex-col max-h-[calc(100vh-10rem)]">`). Dodano `flex-grow overflow-y-auto min-h-0` do `<div id="mainProcessing"` i `<div id="recognizer"`.
*   **Cel:** Umożliwienie kontroli szerokości i wyśrodkowania przez `App.jsx`, oraz poprawne zarządzanie wysokością i przewijaniem wewnętrznym komponentu uploadu.
*   **Zmiana:** Dodano szczegółowe logowanie (za pomocą `onopen`, rozszerzone `onmessage` i `onerror`) do obsługi `EventSource`.
*   **Cel:** Usprawnienie diagnostyki problemów ze strumieniowaniem Server-Sent Events (SSE).

**`frontend/src/index.css`**
*   **Zmiana:** Usunięto globalne style dla `button` (linie 41-51 i 68-70) oraz definicję klasy `.container` (linie 52-59).
*   **Cel:** Usunięcie konfliktów z klasami Tailwind CSS, które powodowały nieprawidłowe wyświetlanie przycisków (np. zawsze biały przycisk "Wyślij" w czacie).
*   **Zmiana:** Usunięto `display: flex; place-items: center;` i `min-height: 100vh;` z reguły `body`. Dodano `height: 100%;` do `html`, `body` i `#root`.
*   **Cel:** Umożliwienie głównemu kontenerowi aplikacji zajmowania pełnej szerokości i zarządzanie ogólną wysokością strony bez przelewania się treści.

**`frontend/vite.config.js`**
*   **Zmiana:** Dla proxy `/process_batch_status/stream` dodano `rewrite: (path) => path`.
*   **Cel:** Zapewnienie, że pełna ścieżka żądania SSE jest przekazywana do backendu bez modyfikacji przez proxy Vite, co było przyczyną błędów 404 na backendzie.
*   **Zmiana:** Usunięto `ws: true` z konfiguracji proxy dla `/process_batch_status/stream`.
*   **Cel:** Zapobieganie potencjalnym konfliktom, ponieważ `ws: true` jest przeznaczone dla WebSocketów, a SSE to standardowe połączenie HTTP.

---

#### Backend (`api_app.py` i `minimal_mcp_server.py`)

**`api_app.py`**
*   **Zmiana:** Dodano import `CORSMiddleware` i skonfigurowano go tak, aby zezwalał na żądania cross-origin z `http://localhost:5173`.
*   **Cel:** Naprawienie problemu blokowania połączeń SSE przez przeglądarkę ze względów bezpieczeństwa (polityka Same-Origin), co było główną przyczyną niedziałającego strumienia statusu.
*   **Zmiana:** Zmieniono poziom logowania z `INFO` na `DEBUG` i dodano bardziej szczegółowe logi w funkcji `process_batch_status_stream`, w tym logowanie `batch_job_info`. Po rozwiązaniu problemu, poziom logowania został przywrócony do `INFO`.
*   **Cel:** Zapewnienie szczegółowej diagnostyki podczas debugowania problemów ze strumieniowaniem SSE.
*   **Zmiana:** Poprawiono formatowanie wiadomości w generatorze `batch_event_generator`, aby używały pojedynczych znaków nowej linii (`\n`) zamiast podwójnych (`\\n`).
*   **Cel:** Zapewnienie zgodności z formatem Server-Sent Events, co jest wymagane do poprawnego parsowania przez `EventSource` w przeglądarce.

**`minimal_mcp_server.py`**
*   **Zmiana:** Usunięto endpoint `/get_qdrant_collections_for_chat/`.
*   **Cel:** Dopasowanie do decyzji o usunięciu selektora kolekcji Qdrant z frontendu.

---
