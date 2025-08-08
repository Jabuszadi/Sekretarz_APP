import tempfile
from pathlib import Path
from fastapi import UploadFile, HTTPException
from processor import convert_mkv_to_wav
# import shutil # No longer needed for manual chunking
import logging
import uuid # Import uuid for generating unique IDs
import os # Import os for path joining

# Dictionary to map job IDs (now file_job_ids or batch_job_ids) to temporary file paths and associated tempdir objects
# This needs to store info for both file and batch jobs.
# For batch jobs, it stores a list of file_job_ids.
# For file jobs, it stores the tempdir path and object.
job_temp_storage = {}

async def save_uploaded_file_temp(file: UploadFile, job_id: str, temp_dir_obj: tempfile.TemporaryDirectory, temp_dir_path: Path) -> Path:
    """
    Save an uploaded file to a temporary location associated with a job_id.
    This function specifically handles saving the file and does NOT perform video conversion.
    It expects the temp_dir_obj and temp_dir_path to be pre-created and passed in.

    Args:
        file (UploadFile): The file uploaded via FastAPI.
        job_id (str): The unique identifier for the job (e.g., speaker label).
        temp_dir_obj (tempfile.TemporaryDirectory): The temporary directory object.
        temp_dir_path (Path): The path to the temporary directory.

    Returns:
        Path: The path to the saved temporary file.
    """
    original_filename = file.filename or f"uploaded_file_{uuid.uuid4().hex}"
    sanitized_filename = "".join([c for c in original_filename if c.isalnum() or c in ('.', '_', '-')]).rstrip('. ')
    if not sanitized_filename:
        sanitized_filename = f"uploaded_file_{uuid.uuid4().hex}"

    saved_file_path = temp_dir_path / sanitized_filename

    try:
        with open(saved_file_path, "wb") as f:
            chunk_size = 8192
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
        logging.info(f"Uploaded file saved to temporary path for job {job_id}: {saved_file_path}")
        return saved_file_path
    except Exception as e:
        logging.error(f"Error saving uploaded file for job {job_id}: {e}")
        temp_dir_obj.cleanup() # Clean up this specific temp dir on failure
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file for job {job_id}: {e}")

async def save_file_to_temp_and_convert_if_needed(file: UploadFile, job_id: str) -> Path:
    """
    Handle file upload, save to a temporary location associated with a job_id (file_job_id),
    and perform initial conversion if needed (e.g., video to WAV).

    Returns:
        Path: The path to the saved (and potentially converted) temporary file.
    """
    # Create a temporary directory specific to this job (file_job_id)
    if job_id in job_temp_storage and 'obj' in job_temp_storage[job_id]:
         logging.warning(f"Temporary directory for job {job_id} already exists in storage. Reusing or overwriting.")

    temp_dir_obj = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir_obj.name)

    if job_id not in job_temp_storage:
        logging.warning(f"Job ID {job_id} not found in job_temp_storage during file upload. Initializing.")
        job_temp_storage[job_id] = {'type': 'file', 'batch_job_id': None, 'status': 'handling_upload'}

    job_temp_storage[job_id]['temp_dir_path'] = temp_dir_path
    job_temp_storage[job_id]['temp_dir_obj'] = temp_dir_obj

    logging.info(f"Created temporary directory for file job {job_id}: {temp_dir_path}")

    # Use the new save_uploaded_file_temp to handle the actual file saving
    original_file_path = await save_uploaded_file_temp(file, job_id, temp_dir_obj, temp_dir_path)

    # Handle video conversion if needed (this part remains the same)
    file_ext = original_file_path.suffix.lower()
    if file_ext in ['.mkv', '.mp4', '.avi', '.mov', '.flv', '.wmv']:
        wav_file_path = temp_dir_path / f"{original_file_path.stem}.wav"
        logging.info(f"Attempting to convert video to WAV for file job {job_id}: {original_file_path} -> {wav_file_path}")
        if not convert_mkv_to_wav(str(original_file_path), str(wav_file_path)):
            logging.error(f"Video conversion failed for file job {job_id}.")
            cleanup_temp_dir(job_id)
            raise HTTPException(status_code=500, detail=f"Video conversion failed for file job {job_id}")
        logging.info(f"Video conversion successful for file job {job_id}.")
        return wav_file_path

    logging.info(f"No video conversion needed for file job {job_id}.")
    return original_file_path


def cleanup_temp_dir(job_id: str | None = None):
    """
    Clean up temporary directory associated with a specific job_id (file_job_id).
    Does NOT clean up batch job entries.
    """
    global job_temp_storage
    # We only clean up directories associated with 'file' type jobs
    if job_id and job_id in job_temp_storage and job_temp_storage[job_id].get('type') == 'file':
        try:
            # Retrieve and remove the temp dir object for this file job
            temp_info = job_temp_storage.get(job_id)
            if temp_info and 'temp_dir_obj' in temp_info:
                temp_dir_obj = temp_info['temp_dir_obj']
                temp_dir_obj.cleanup() # Clean up the directory
                logging.info(f"Temporary directory cleaned up for file job {job_id}.")
            else:
                 logging.warning(f"No temporary directory object found for file job {job_id} to clean up.")

            # We don't remove the job_id entry itself here, as the batch process manages that.
            # The batch process will remove the file_job_id entries (or the batch entry) when done.

        except Exception as e:
            # This might happen if files inside are still in use
            logging.warning(f"Could not clean up temporary directory for file job {job_id}: {e}")
    elif job_id is None:
         logging.warning("cleanup_temp_dir called without a job_id. No specific temp dir cleaned.")
    elif job_id in job_temp_storage and job_temp_storage[job_id].get('type') == 'batch':
         logging.debug(f"Attempted to clean up temp dir for batch job {job_id}. Skipping - batch cleanup is different.")
    else:
        logging.warning(f"cleanup_temp_dir called with unknown or already cleaned job_id: {job_id}")


# You will need to update your api_app.py to call cleanup_temp_dir(file_job_id)
# at the end of processing EACH file in the batch.
# The batch process will remove the file job entries and the batch entry from job_temp_storage.