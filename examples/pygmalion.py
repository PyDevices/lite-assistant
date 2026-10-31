# Usage:  python pygmalion.py ~/pygmalion_wav --brackets [split | keep | remove]
# Save https://www.gutenberg.org/cache/epub/3825/pg3825.txt in the pygmalion.py directory
import os
import re
import argparse

character_mapping = {
    "A BYSTANDER": {"model": "kokoro", "voice": "am_santa"},
    "A SARCASTIC BYSTANDER": {"model": "kokoro", "voice": "am_adam"},
    "CLARA": {"model": "kokoro", "voice": "bf_alice"},
    "DOOLITTLE": {"model": "kokoro", "voice": "bm_daniel"},
    "FREDDY": {"model": "kokoro", "voice": "bm_fable"},
    "HIGGINS": {"model": "kokoro", "voice": "bm_george"},
    "LIZA": {"model": "kokoro", "voice": "bf_emma"},
    "MISS EYNSFORD HILL": {"model": "kokoro", "voice": "bf_kore"},
    "MRS. EYNSFORD HILL": {"model": "kokoro", "voice": "bf_lily"},
    "MRS. HIGGINS": {"model": "kokoro", "voice": "bf_isabella"},
    "MRS. PEARCE": {"model": "kokoro", "voice": "af_nicole"},
    "PICKERING": {"model": "kokoro", "voice": "bm_lewis"},
    "SEVERAL BYSTANDERS": {"model": "kokoro", "voice": "am_fenrir"},
    "THE BYSTANDER": {"model": "kokoro", "voice": "pm_santa"},
    "THE BYSTANDERS GENERALLY": {"model": "kokoro", "voice": "am_onyx"},
    "THE DAUGHTER": {"model": "kokoro", "voice": "bf_emma"},
    "THE FLOWER GIRL": {"model": "kokoro", "voice": "af_heart"},
    "THE GENTLEMAN": {"model": "kokoro", "voice": "am_eric"},
    "THE GENTLEMEN": {"model": "kokoro", "voice": "am_liam"},
    "THE JAPANESE LADY": {"model": "kokoro", "voice": "zf_xiaobei"},
    "THE MOTHER": {"model": "kokoro", "voice": "bf_lily"},
    "THE NOTE TAKER": {"model": "kokoro", "voice": "bm_george"},
    "THE SARCASTIC BYSTANDER": {"model": "kokoro", "voice": "am_adam"},
    "THE SARCASTIC ONE": {"model": "kokoro", "voice": "am_puck"},
    "NARRATOR": {"model": "kokoro", "voice": "am_michael"},
}


def process_script_file(filepath, output_dir=".", server_ip="127.0.0.1", port=5000, bracket_mode="remove"):
  from tts_client import save_to_wav

  os.makedirs(output_dir, exist_ok=True)

  with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

  acts = re.split(r"\n+(ACT\s+[IVXLCDM]+)\n+", content)

  current_act = "act_0_preface"
  act_segments = {current_act: []}

  for i in range(1, len(acts), 2):
    act_title_clean = acts[i].strip().lower().replace(" ", "_")
    current_act = act_title_clean
    act_segments[current_act] = []
    body = acts[i + 1]

    paragraphs = re.split(r"\n\s*\n", body)

    narrator_buffer = []

    def flush_narrator():
      if narrator_buffer:
        nar_text = " ".join(narrator_buffer).strip()
        if len(nar_text) > 2:
          act_segments[current_act].append(("NARRATOR", nar_text))
        narrator_buffer.clear()

    for para in paragraphs:
      para_clean = re.sub(r"\s+", " ", para).strip()
      if not para_clean:
        continue

      # Updated regex: captures speaker name, then captures everything following (including brackets) into dialogue_text
      match = re.match(r"^((?:[A-Z]+\.)?\s*[A-Z\s\.]+?)(?=\s*\[|\.\s+)(.*)$", para_clean)

      if match:
        speaker = match.group(1).strip()
        if speaker in character_mapping:
          flush_narrator()
          dialogue_text = match.group(2).strip()
          # Clean up leading period/spaces from dialogue match if present
          dialogue_text = re.sub(r"^\.?\s*", "", dialogue_text)

          if bracket_mode == "remove":
            clean_dialogue = re.sub(r"\s*\[.*?\]\s*", " ", dialogue_text).strip()
            if clean_dialogue:
              act_segments[current_act].append((speaker, clean_dialogue))

          elif bracket_mode == "keep":
            act_segments[current_act].append((speaker, dialogue_text))

          elif bracket_mode == "split":
            parts = re.split(r"(\[.*?\])", dialogue_text)
            for part in parts:
              part_str = part.strip()
              if not part_str:
                continue
              if part_str.startswith("[") and part_str.endswith("]"):
                stage_dir = part_str[1:-1].strip()
                if stage_dir:
                  act_segments[current_act].append(("NARRATOR", stage_dir))
              else:
                act_segments[current_act].append((speaker, part_str))
        else:
          narrator_buffer.append(para_clean)
      else:
        narrator_buffer.append(para_clean)

    flush_narrator()

  for act_name, segments in act_segments.items():
    if not segments:
      continue
    print(
        f"\nProcessing {act_name} ({len(segments)} segments) into individual"
        f" WAV files using bracket mode: '{bracket_mode}'..."
    )

    act_sub_dir = os.path.join(output_dir, act_name)
    os.makedirs(act_sub_dir, exist_ok=True)

    for idx, (speaker, text) in enumerate(segments):
      config = character_mapping.get(
          speaker, {"model": "kokoro", "voice": "af_heart"}
      )
      model_name = config["model"]
      voice_id = config["voice"]

      segment_filename = os.path.join(
          act_sub_dir, f"{act_name}_seg_{idx:04d}_{speaker.lower().replace(' ', '_')}.wav"
      )
      print(f"[{speaker}] {text}")

      try:
        save_to_wav(
            segment_filename,
            text,
            voice=voice_id,
            model=model_name,
            server_ip=server_ip,
            port=port
        )
        print(f"-> File {segment_filename} wrote using voice '{voice_id}' from '{model_name}'")
      except Exception as e:
        print(f"Error generating segment {idx} for {speaker}: {e}")

    print(f"Completed all segments for {act_name} in folder: {act_sub_dir}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Process Pygmalion script into individual line WAV files with flexible bracket handling."
  )
  parser.add_argument("filepath", help="Path to the script text file (e.g., pg3825.txt)")
  parser.add_argument(
      "-o",
      "--output-dir",
      default=".",
      help="Directory to save output Act folders (default: current directory '.')",
  )
  parser.add_argument(
      "--brackets",
      choices=["remove", "keep", "split"],
      default="remove",
      help="How to handle inline bracketed stage directions ('remove', 'keep', or 'split' to narrator)",
  )
  parser.add_argument("--host", default="127.0.0.1", help="TTS server IP address")
  parser.add_argument("--port", type=int, default=5000, help="TTS server port")

  args = parser.parse_args()

  script_path = args.filepath
  output_dir = args.output_dir

  if os.path.isdir(script_path):
    candidate = os.path.join(script_path, "pg3825.txt")
    if os.path.isfile(candidate):
      script_path = candidate
    else:
      if output_dir == ".":
        output_dir = args.filepath
        script_path = "pg3825.txt"

  process_script_file(
      script_path,
      output_dir=output_dir,
      server_ip=args.host,
      port=args.port,
      bracket_mode=args.brackets,
  )
