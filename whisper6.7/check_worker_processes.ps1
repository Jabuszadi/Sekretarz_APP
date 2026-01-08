# Skrypt PowerShell do sprawdzania procesów Dramatiq workera
Write-Host "=" * 80
Write-Host "Sprawdzanie procesów Dramatiq workera"
Write-Host "=" * 80

# Znajdź wszystkie procesy Python, które mogą być workerami Dramatiq
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*dramatiq*" -or $_.CommandLine -like "*queue_service*"
}

if ($pythonProcesses) {
    Write-Host "`nZnaleziono procesy Python związane z Dramatiq:"
    foreach ($proc in $pythonProcesses) {
        Write-Host "  - PID: $($proc.Id), Nazwa: $($proc.ProcessName)"
        try {
            $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
            Write-Host "    Komenda: $cmdLine"
        } catch {
            Write-Host "    (Nie można odczytać komendy)"
        }
    }
    Write-Host "`nAby zakończyć proces, użyj:"
    Write-Host "  Stop-Process -Id <PID> -Force"
} else {
    Write-Host "`nNie znaleziono procesów Dramatiq workera."
}

# Sprawdź też procesy uvicorn (API)
$uvicornProcesses = Get-Process python -ErrorAction SilentlyContinue | Where-Object {
    $_.CommandLine -like "*uvicorn*" -or $_.CommandLine -like "*api_app*"
}

if ($uvicornProcesses) {
    Write-Host "`nZnaleziono procesy uvicorn (API):"
    foreach ($proc in $uvicornProcesses) {
        Write-Host "  - PID: $($proc.Id), Nazwa: $($proc.ProcessName)"
    }
}

Write-Host "`n" + "=" * 80
