import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_current_vram_usage_gb():
    """
    Pobiera bieżące zużycie VRAM (pamięci karty graficznej) w GB,
    używając nvidia-smi (tylko dla kart NVIDIA).
    """
    try:
        command = 'nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits'
        result = subprocess.run(command, capture_output=True, text=True, shell=True, check=True)
        
        vram_mib = float(result.stdout.strip())
        vram_gb = vram_mib / 1024
        return vram_gb
    except FileNotFoundError:
        logging.warning("nvidia-smi not found. VRAM usage cannot be retrieved. Ensure NVIDIA drivers are installed.")
    except subprocess.CalledProcessError as e:
        logging.error(f"nvidia-smi command failed with error: {e.stderr}")
    except ValueError:
        logging.error(f"Could not parse VRAM usage from nvidia-smi output: '{result.stdout.strip()}'")
    except Exception as e:
        logging.error(f"An unexpected error occurred while getting VRAM usage: {e}")
    return 0.0

if __name__ == "__main__":
    vram_usage = get_current_vram_usage_gb()
    logging.info(f"Test VRAM Usage: {vram_usage:.2f} GB")
    if vram_usage > 0:
        logging.info("VRAM usage successfully retrieved and is greater than 0.")
    else:
        logging.warning("VRAM usage is 0 or could not be retrieved. Check for errors above or if NVIDIA GPU is present.")
