import os
import re
import argparse


def extract_unique_speakers(filepath):
  with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

  # Split script into Acts using headers like "ACT I"
  acts = re.split(r"\n+(ACT\s+[IVXLCDM]+)\n+", content)

  unique_speakers = set()

  for i in range(1, len(acts), 2):
    body = acts[i + 1]
    paragraphs = re.split(r"\n\s*\n", body)

    for para in paragraphs:
      para_clean = re.sub(r"\s+", " ", para).strip()
      if not para_clean:
        continue

      # Use our robust regex to catch speaker names
      # Updated regex to allow periods within the speaker name (e.g., "MRS. HIGGINS")
      match = re.match(
        r"^((?:[A-Z]+\.)?\s*[A-Z\s\.]+?)(?=\s*\[|\.\s+)(?:\s*\[.*?\])?\.?\s+(.*)$",
        para_clean,
      )

      if match:
        speaker = match.group(1).strip()
        unique_speakers.add(speaker)

  print(f"Found {len(unique_speakers)} unique speaker names:")
  for spk in sorted(unique_speakers):
    print(f"  - '{spk}'")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Extract all unique speaker names from Pygmalion script.")
  parser.add_argument("filepath", help="Path to the script text file (e.g., pg3825.txt)")
  args = parser.parse_args()

  extract_unique_speakers(args.filepath)
