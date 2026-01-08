# Skrypt PowerShell do sprawdzania procesów Dramatiq workera
Write-Host "=" * 80
Write-Host "Sprawdzanie procesow Dramatiq workera"
Write-Host "=" * 80

# Znajdź wszystkie procesy Python
$pythonProcesses = Get-Process python -ErrorAction SilentlyContinue

if ($pythonProcesses) {
    Write-Host "`nZnaleziono procesy Python:"
    $dramatiqFound = $false
    
    foreach ($proc in $pythonProcesses) {
        try {
            $cmdLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($proc.Id)").CommandLine
            if ($cmdLine -like "*dramatiq*" -or $cmdLine -like "*queue_service*") {
                $dramatiqFound = $true
                Write-Host "  [DRAMATIQ WORKER] PID: $($proc.Id)"
                Write-Host "    Komenda: $cmdLine"
            } elseif ($cmdLine -like "*uvicorn*" -or $cmdLine -like "*api_app*") {
                Write-Host "  [API] PID: $($proc.Id)"
                Write-Host "    Komenda: $cmdLine"
            } else {
                Write-Host "  [OTHER] PID: $($proc.Id)"
                Write-Host "    Komenda: $cmdLine"
            }
        } catch {
            Write-Host "  [UNKNOWN] PID: $($proc.Id) (nie mozna odczytac komendy)"
        }
    }
    
    if (-not $dramatiqFound) {
        Write-Host "`n[WARN] Nie znaleziono procesu Dramatiq workera!"
        Write-Host "   Uruchom worker w osobnym terminalu:"
        Write-Host "   python -m dramatiq queue_service --queues chat,batch_processing,file_processing"
    } else {
        Write-Host "`n[OK] Znaleziono proces Dramatiq workera"
    }
} else {
    Write-Host "`n[WARN] Nie znaleziono zadnych procesow Python!"
    Write-Host "   Uruchom worker:"
    Write-Host "   python -m dramatiq queue_service --queues chat,batch_processing,file_processing"
}

Write-Host "`n" + "=" * 80
