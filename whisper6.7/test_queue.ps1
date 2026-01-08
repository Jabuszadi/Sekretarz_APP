# Skrypt PowerShell do testowania kolejki (dla Windows)

$API_URL = "http://localhost:8000"
$jobIds = @()

Write-Host "🚀 Test systemu kolejkowania - wysyłanie 5 zapytań..." -ForegroundColor Green
Write-Host ""

# Wyślij 5 zapytań
for ($i = 1; $i -le 5; $i++) {
    Write-Host "📤 Wysyłanie zapytania $i..." -ForegroundColor Yellow
    
    $body = @{
        query = "Test zapytanie numer $i"
    } | ConvertTo-Json
    
    try {
        $response = Invoke-RestMethod -Uri "$API_URL/chat/query" `
            -Method POST `
            -ContentType "application/json" `
            -Body $body
        
        $responseText = $response.response
        if ($responseText -match "ID zadania: ([a-f0-9-]+)") {
            $jobId = $matches[1]
            Write-Host "✅ Job ID: $jobId" -ForegroundColor Green
            $jobIds += $jobId
        } else {
            Write-Host "❌ Błąd: Nie znaleziono job_id w odpowiedzi" -ForegroundColor Red
            Write-Host "   Odpowiedź: $responseText" -ForegroundColor Gray
        }
    } catch {
        Write-Host "❌ Błąd: $_" -ForegroundColor Red
    }
    Write-Host ""
}

Write-Host "⏳ Czekam 10 sekund przed sprawdzeniem statusu..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host ""
Write-Host "📊 Sprawdzanie statusu zadań..." -ForegroundColor Green
Write-Host ""

# Sprawdź status każdego zadania
foreach ($jobId in $jobIds) {
    Write-Host "Job $jobId:" -ForegroundColor Cyan
    try {
        $status = Invoke-RestMethod -Uri "$API_URL/chat/query/status/$jobId" -Method GET
        Write-Host "  Status: $($status.status)" -ForegroundColor $(if ($status.status -eq "completed") { "Green" } elseif ($status.status -eq "failed") { "Red" } else { "Yellow" })
        Write-Host "  Próba: $($status.retry_count + 1)/$($status.max_retries)" -ForegroundColor Gray
        
        if ($status.status -eq "completed") {
            $response = Invoke-RestMethod -Uri "$API_URL/chat/query/response/$jobId" -Method GET
            $preview = if ($response.response.Length -gt 100) { 
                $response.response.Substring(0, 100) + "..." 
            } else { 
                $response.response 
            }
            Write-Host "  Odpowiedź: $preview" -ForegroundColor Gray
        } elseif ($status.status -eq "failed") {
            Write-Host "  Błąd: $($status.error_message)" -ForegroundColor Red
        }
    } catch {
        Write-Host "  ❌ Błąd przy sprawdzaniu statusu: $_" -ForegroundColor Red
    }
    Write-Host ""
}

Write-Host "✅ Test zakończony" -ForegroundColor Green

