#!/usr/bin/env python3
"""
Skrypt do monitorowania kolejek Dramatiq w czasie rzeczywistym.
Pokazuje statystyki kolejki i listę aktywnych zadań.
"""
import asyncio
import os
import sys
import time
from typing import Dict, Any
import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:7777")


def clear_screen():
    """Czyści ekran terminala."""
    os.system('cls' if os.name == 'nt' else 'clear')


def format_stats(stats: Dict[str, Any]) -> str:
    """Formatuje statystyki do wyświetlenia."""
    lines = []
    lines.append("=" * 60)
    lines.append("📊 STATYSTYKI KOLEJKI DRAMATIQ")
    lines.append("=" * 60)
    
    if stats.get("error"):
        lines.append(f"❌ Błąd: {stats.get('error')}")
        return "\n".join(lines)
    
    redis_status = "✅ Połączono" if stats.get("redis_connected") else "❌ Rozłączono"
    lines.append(f"Redis: {redis_status}")
    lines.append("")
    
    lines.append(f"⏳ Oczekujące:     {stats.get('pending', 0):>6}")
    lines.append(f"✅ Zakończone:     {stats.get('completed', 0):>6}")
    lines.append(f"❌ Błędy:          {stats.get('failed', 0):>6}")
    lines.append(f"📊 Razem:          {stats.get('total', 0):>6}")
    lines.append(f"📦 Rozmiar kolejki: {stats.get('queue_size', 0):>6}")
    
    return "\n".join(lines)


def format_jobs(jobs: list) -> str:
    """Formatuje listę zadań do wyświetlenia."""
    if not jobs:
        return "📋 Brak aktywnych zadań w kolejce"
    
    lines = []
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"📋 AKTYWNE ZADANIA ({len(jobs)})")
    lines.append("=" * 60)
    
    for i, job in enumerate(jobs, 1):
        job_id = job.get("job_id", "unknown")[:16] + "..."
        status = job.get("status", "unknown")
        status_emoji = {
            "pending": "⏳",
            "delayed": "⏰",
            "processing": "🔄",
            "completed": "✅",
            "failed": "❌",
        }.get(status, "❓")
        
        lines.append(f"{i:>3}. {status_emoji} {job_id} | Status: {status}")
        if job.get("scheduled_at"):
            lines.append(f"     Zaplanowane na: {job.get('scheduled_at')}")
    
    return "\n".join(lines)


async def fetch_stats() -> Dict[str, Any]:
    """Pobiera statystyki z API."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_URL}/queue/stats")
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
    except httpx.ConnectError:
        return {"error": "Nie można połączyć się z API. Upewnij się, że serwer działa na porcie 7777."}
    except Exception as e:
        return {"error": str(e)}


async def fetch_jobs(limit: int = 20) -> list:
    """Pobiera listę aktywnych zadań z API."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{API_URL}/queue/jobs", params={"limit": limit})
            if response.status_code == 200:
                data = response.json()
                return data.get("jobs", [])
            else:
                return []
    except Exception as e:
        return []


async def monitor_loop(interval: float = 2.0, show_jobs: bool = True, jobs_limit: int = 20):
    """Główna pętla monitorowania."""
    print("🚀 Uruchamianie monitora kolejek...")
    print(f"📡 Łączenie z API: {API_URL}")
    print(f"⏱️  Interwał odświeżania: {interval}s")
    print("\nNaciśnij CTRL+C, aby zakończyć\n")
    
    try:
        while True:
            # Pobierz statystyki i zadania równolegle
            if show_jobs:
                stats, jobs = await asyncio.gather(
                    fetch_stats(),
                    fetch_jobs(jobs_limit)
                )
            else:
                stats = await fetch_stats()
                jobs = []
            
            # Wyczyść ekran i wyświetl dane
            clear_screen()
            print(format_stats(stats))
            
            if show_jobs:
                print(format_jobs(jobs))
            
            print("\n" + "=" * 60)
            print(f"🕐 Ostatnie odświeżenie: {time.strftime('%H:%M:%S')}")
            print("Naciśnij CTRL+C, aby zakończyć")
            
            await asyncio.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Zakończono monitorowanie.")


def main():
    """Główna funkcja."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Monitor kolejek Dramatiq",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  python monitor_queue.py                    # Monitor z domyślnymi ustawieniami
  python monitor_queue.py -i 5                # Odświeżaj co 5 sekund
  python monitor_queue.py --no-jobs           # Tylko statystyki, bez listy zadań
  python monitor_queue.py -j 50               # Pokaż do 50 zadań
        """
    )
    
    parser.add_argument(
        "-i", "--interval",
        type=float,
        default=2.0,
        help="Interwał odświeżania w sekundach (domyślnie: 2.0)"
    )
    
    parser.add_argument(
        "--no-jobs",
        action="store_true",
        help="Nie pokazuj listy aktywnych zadań"
    )
    
    parser.add_argument(
        "-j", "--jobs-limit",
        type=int,
        default=20,
        help="Maksymalna liczba zadań do wyświetlenia (domyślnie: 20)"
    )
    
    parser.add_argument(
        "--api-url",
        type=str,
        default=None,
        help=f"URL API (domyślnie: {API_URL})"
    )
    
    args = parser.parse_args()
    
    global API_URL
    if args.api_url:
        API_URL = args.api_url
    
    try:
        asyncio.run(monitor_loop(
            interval=args.interval,
            show_jobs=not args.no_jobs,
            jobs_limit=args.jobs_limit
        ))
    except KeyboardInterrupt:
        print("\n👋 Zakończono.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Błąd: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
