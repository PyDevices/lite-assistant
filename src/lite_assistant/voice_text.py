"""
Text handling for the /v1/voice endpoint: splitting a streamed LLM reply into
sentences for TTS, and cleaning each sentence before it is spoken.
"""
import re

VOICE_SYSTEM_PROMPT = ("You are a helpful voice assistant. Your replies are spoken aloud, so answer in "
                       "a few short, plain sentences, with no markdown, lists, or emoji.")

# A sentence ends at . ! or ? (plus any closing quote or bracket) followed by
# whitespace, or at a line break
SENTENCE_END = re.compile(r'(?<=[.!?])["\')\]]*\s+|\n+')

def split_sentences(chunks):
    """ Regroups a stream of text chunks into sentences, yielding each one as soon as it ends. """
    buffer = ""
    for chunk in chunks:
        buffer += chunk
        while (match := SENTENCE_END.search(buffer)):
            sentence, buffer = buffer[:match.end()].strip(), buffer[match.end():]
            if sentence:
                yield sentence
    if buffer.strip():
        yield buffer.strip()

def speakable(sentence):
    """ Drops markdown the model slips in anyway, so TTS doesn't read it out. """
    return re.sub(r'^\s*(?:[-*]|\d+\.)\s+|[*#`_~]', '', sentence).strip()
