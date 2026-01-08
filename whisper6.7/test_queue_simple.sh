#!/bin/bash
# Prosty skrypt bash do testowania kolejki (dla Linux/Mac)

API_URL="http://localhost:8000"

echo "🚀 Test systemu kolejkowania - wysyłanie 5 zapytań..."
echo ""

# Wyślij 5 zapytań
for i in {1..5}; do
    echo "📤 Wysyłanie zapytania $i..."
    response=$(curl -s -X POST "$API_URL/chat/query" \
        -H "Content-Type: application/json" \
        -d "{\"query\": \"Test zapytanie numer $i\"}")
    
    job_id=$(echo $response | grep -oP 'ID zadania: \K[a-f0-9-]+' || echo "")
    
    if [ -n "$job_id" ]; then
        echo "✅ Job ID: $job_id"
        echo "$job_id" >> job_ids.txt
    else
        echo "❌ Błąd: $response"
    fi
    echo ""
done

echo "⏳ Czekam 10 sekund przed sprawdzeniem statusu..."
sleep 10

echo ""
echo "📊 Sprawdzanie statusu zadań..."
echo ""

# Sprawdź status każdego zadania
while IFS= read -r job_id; do
    if [ -n "$job_id" ]; then
        echo "Job $job_id:"
        curl -s "$API_URL/chat/query/status/$job_id" | python3 -m json.tool
        echo ""
    fi
done < job_ids.txt

# Usuń plik tymczasowy
rm -f job_ids.txt

echo "✅ Test zakończony"

