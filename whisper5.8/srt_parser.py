# srt_parser.py
import re
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

def parse_srt_files(folder_path: str) -> pd.DataFrame:
    """Parses all SRT files in a folder into a pandas DataFrame."""
    folder = Path(folder_path)
    if not folder.is_dir():
        logging.error(f"❌ SRT source folder not found: {folder_path}")
        return pd.DataFrame() # Return empty DataFrame

    srt_files = list(folder.glob("*.srt"))
    if not srt_files:
        logging.warning(f"⚠️ No SRT files found in: {folder_path}")
        return pd.DataFrame()

    logging.info(f"🔍 Found {len(srt_files)} SRT files in {folder_path}. Parsing...")

    # Adjusted pattern to be more robust, assuming speaker might be missing or format varies
    # This regex assumes: float_seconds float_seconds speaker_token content
    # Example: 0.66 3.27 SPEAKER1 some text here
    # It might need adjustment based on the exact SRT format from Alchemist
    line_pattern = re.compile(r'(\d+\.\d+)\s+(\d+\.\d+)\s+([^\s]+)\s+(.*)')
    # Alternative if speaker token is sometimes missing or part of content:
    # line_pattern = re.compile(r'(\d+\.\d+)\s+(\d+\.\d+)\s+(.*)') # Simpler, gets all text after times

    all_data = []
    files_parsed_count = 0
    files_skipped_count = 0

    for file_path in srt_files:
        try:
            with open(file_path, encoding='utf-8') as f:
                raw_text = f.read()
            file_name = file_path.name

            # Extract date and time from filename (YYYY-MM-DD HH-MM-SS format expected)
            date_str_match = re.match(r"(\d{4}-\d{2}-\d{2}\s\d{2}-\d{2}-\d{2})", file_name)
            if not date_str_match:
                logging.warning(f"⚠️ Skipping file '{file_name}': Could not extract date prefix (YYYY-MM-DD HH-MM-SS).")
                files_skipped_count += 1
                continue

            date_str = date_str_match.group(1)
            try:
                file_date = datetime.strptime(date_str, "%Y-%m-%d %H-%M-%S")
            except ValueError:
                logging.warning(f"⚠️ Skipping file '{file_name}': Invalid date format in extracted prefix '{date_str}'.")
                files_skipped_count += 1
                continue

            lines_parsed_in_file = 0
            for line in raw_text.strip().split('\n'):
                match = line_pattern.match(line)
                if match:
                    # Adapt based on the chosen regex pattern
                    from_time, to_time, speaker, content = match.groups() # Assumes 4 groups
                    # If using the simpler regex:
                    # from_time, to_time, content_with_maybe_speaker = match.groups()
                    # speaker = "UNKNOWN" # Or further processing to extract speaker if possible

                    all_data.append({
                        'from_time': float(from_time),
                        'to_time': float(to_time),
                        'speaker': speaker.strip(), # Ensure speaker tag has no extra spaces
                        'content': content.strip(),
                        'file_path': str(file_path), # Store full path for grouping later
                        'date': file_date # Store datetime object
                    })
                    lines_parsed_in_file += 1
                # else: # Optional: log lines that don't match the pattern
                #     logging.debug(f"  Line not matched in {file_name}: '{line}'")

            if lines_parsed_in_file > 0:
                logging.debug(f"  Parsed {lines_parsed_in_file} lines from '{file_name}'.")
                files_parsed_count += 1
            else:
                logging.warning(f"⚠️ No matching lines found in file '{file_name}'. It might be empty or have an unexpected format.")
                files_skipped_count +=1


        except Exception as e:
            logging.error(f"❌ Error processing file {file_path}: {e}")
            files_skipped_count += 1
            continue # Skip to next file on error

    if not all_data:
        logging.warning("⚠️ No data extracted from any SRT files.")
        return pd.DataFrame()

    df = pd.DataFrame(all_data)
    # df.index += 1 # Index starting from 1 might not be necessary
    logging.info(f"✅ Successfully parsed {files_parsed_count} files ({files_skipped_count} skipped). Total lines extracted: {len(df)}")
    return df