"""
Prosty skrypt do testowania czy endpoint /chat/query jest dostępny.
"""
import requests
import json

API_URL = "http://localhost:7777"

print("🔍 Sprawdzanie dostępności endpointów...\n")

# Test 1: Health check
try:
    response = requests.get(f"{API_URL}/health", timeout=5)
    print(f"✅ /health: {response.status_code}")
    if response.status_code == 200:
        print(f"   Odpowiedź: {response.json()}")
except Exception as e:
    print(f"❌ /health: Błąd - {e}")

# Test 2: Sprawdź czy endpoint /chat/query istnieje
try:
    response = requests.post(
        f"{API_URL}/chat/query",
        json={"query": "test"},
        headers={"Content-Type": "application/json"},
        timeout=5
    )
    print(f"\n📤 POST /chat/query: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   Odpowiedź: {json.dumps(data, indent=2, ensure_ascii=False)}")
    elif response.status_code == 404:
        print(f"   ❌ Endpoint nie znaleziony! Sprawdź czy serwer został zrestartowany.")
    else:
        print(f"   Błąd: {response.text[:200]}")
except requests.exceptions.ConnectionError:
    print(f"\n❌ Nie można połączyć się z {API_URL}")
    print("   Upewnij się, że serwer jest uruchomiony: python run.py")
except Exception as e:
    print(f"\n❌ Błąd: {e}")

# Test 3: Sprawdź dostępne endpointy (jeśli FastAPI ma /docs)
try:
    response = requests.get(f"{API_URL}/docs", timeout=5)
    if response.status_code == 200:
        print(f"\n✅ Dokumentacja API dostępna: {API_URL}/docs")
except:
    pass

print("\n💡 Jeśli endpoint zwraca 404:")
print("   1. Upewnij się, że serwer został zrestartowany po dodaniu endpointów")
print("   2. Sprawdź logi serwera czy są błędy podczas startu")
print("   3. Sprawdź czy pgBoss został poprawnie zainicjalizowany")

