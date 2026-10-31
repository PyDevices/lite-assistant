"""
Unit tests for the sentence splitting and cleanup behind /v1/voice.
"""
import unittest

from lite_assistant.voice_text import split_sentences, speakable

class TestSplitSentences(unittest.TestCase):
    def test_sentences_split_across_chunks(self):
        chunks = ["Hel", "lo there. How", " are you?", " Fine!"]
        self.assertEqual(list(split_sentences(chunks)), ["Hello there.", "How are you?", "Fine!"])

    def test_sentence_is_yielded_before_the_stream_ends(self):
        def chunks():
            yield "First sentence. Sec"
            raise AssertionError("read past the first sentence before yielding it")
        self.assertEqual(next(split_sentences(chunks())), "First sentence.")

    def test_line_break_ends_a_sentence(self):
        self.assertEqual(list(split_sentences(["One\nTwo\n\nThree"])), ["One", "Two", "Three"])

    def test_closing_quote_stays_with_its_sentence(self):
        self.assertEqual(list(split_sentences(['She said "hi." Then left.'])),
                         ['She said "hi."', "Then left."])

    def test_decimal_point_does_not_split(self):
        self.assertEqual(list(split_sentences(["Pi is 3.14 or so."])), ["Pi is 3.14 or so."])

    def test_unterminated_tail_is_flushed(self):
        self.assertEqual(list(split_sentences(["Done. And then"])), ["Done.", "And then"])

    def test_empty_stream(self):
        self.assertEqual(list(split_sentences([])), [])
        self.assertEqual(list(split_sentences(["  ", "\n"])), [])

class TestSpeakable(unittest.TestCase):
    def test_strips_markdown(self):
        self.assertEqual(speakable("**Bold** and `code`."), "Bold and code.")
        self.assertEqual(speakable("## Heading"), "Heading")

    def test_strips_list_markers(self):
        self.assertEqual(speakable("- item one"), "item one")
        self.assertEqual(speakable("* item two"), "item two")
        self.assertEqual(speakable("1. First"), "First")

    def test_plain_text_unchanged(self):
        self.assertEqual(speakable("It costs 3.50 dollars."), "It costs 3.50 dollars.")

    def test_markdown_only_becomes_empty(self):
        self.assertEqual(speakable("***"), "")

if __name__ == '__main__':
    unittest.main()
