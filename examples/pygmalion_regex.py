import os
import re
import argparse


def debug_parse_script(filepath):
  with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

  # Focus on Act I for testing
  acts = re.split(r"\n+(ACT\s+[IVXLCDM]+)\n+", content)
  body = ""
  for i in range(1, len(acts), 2):
    if acts[i].strip() == "ACT I":
      body = acts[i + 1]
      break

  # Split into paragraphs using double newlines or multiple blank lines
  paragraphs = re.split(r"\n\s*\n", body)

  print(f"Total paragraphs found in Act I: {len(paragraphs)}\n")
  print("-" * 60)

  for idx, para in enumerate(paragraphs[:15]):  # Inspect first 15 paragraphs
    # Normalize internal newlines and spaces
    para_clean = re.sub(r"\s+", " ", para).strip()
    if not para_clean:
      continue

    # Test our standard regex pattern
    match = re.match(
      r"^([A-Z\s]+?)(?=\s*\[|\.\s+)(?:\s*\[.*?\])?\.?\s+(.*)$", para_clean
    )

    if match:
      speaker = match.group(1).strip()
      text_snippet = match.group(2)[:50] + "..."
      print(f"[{idx}] MATCHED SPEAKER: '{speaker}'")
      print(f"    Text: {text_snippet}")
    else:
      snippet = para_clean[:80] + "..." if len(para_clean) > 80 else para_clean
      print(f"[{idx}] NARRATOR / UNMATCHED block:")
      print(f"    Snippet: {snippet}")
    print("-" * 60)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="Debug Pygmalion script parsing.")
  parser.add_argument("filepath", help="Path to the script text file (e.g., pg3825.txt)")
  args = parser.parse_args()

  debug_parse_script(args.filepath)
