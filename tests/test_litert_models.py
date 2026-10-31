"""
Tests for finding, adding and removing LiteRT-LM models, and the 'lite-assistant
models' command, with the Hugging Face cache and Hub faked.
"""
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from lite_assistant import cli, litert_models

GB = 10**9
TEXT = {"modalities": ["text"], "context": 4096, "speculative": False}


def hub_repo(repo_id, files, tags=(), downloads=0):
    """ What HfApi's list_models and model_info say about a repo: files maps names to sizes. """
    return SimpleNamespace(id=repo_id, tags=list(tags), downloads=downloads, likes=0,
                           siblings=[SimpleNamespace(rfilename=name, size=size) for name, size in files.items()])


class FakeHub:
    def __init__(self, *repos):
        self.repos = {repo.id: repo for repo in repos}

    def list_models(self, filter=None, search=None, sort=None, limit=None):
        assert filter == litert_models.LIBRARY
        self.sort = sort
        return [r for r in self.repos.values() if not search or search.lower() in r.id.lower()][:limit]

    def model_info(self, repo_id, files_metadata=False):
        from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
        if repo_id.startswith("gated/"):
            raise GatedRepoError("gated", response=mock.Mock(status_code=401))
        if repo_id not in self.repos:
            raise RepositoryNotFoundError("missing", response=mock.Mock(status_code=404))
        return self.repos[repo_id]


class Cache:
    """ A Hugging Face cache in a temporary folder: snapshot links to blobs, as hf_hub_download leaves them. """
    def __init__(self):
        self.folder = tempfile.TemporaryDirectory()
        self.repos = []

    def add(self, repo_id, name, size=10, modified=1):
        root = Path(self.folder.name) / repo_id.replace("/", "--")
        blob = root / "blobs" / f"{name}-{modified}"
        link = root / "snapshots" / str(modified) / name
        blob.parent.mkdir(parents=True, exist_ok=True)
        link.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(b"x" * size)
        link.symlink_to(blob)
        file = SimpleNamespace(file_name=name, file_path=link, blob_path=blob, size_on_disk=size)
        repo = next((r for r in self.repos if r.repo_id == repo_id), None)
        if not repo:
            repo = SimpleNamespace(repo_id=repo_id, repo_type="model", revisions=[], repo_path=root)
            self.repos.append(repo)
        repo.revisions.append(SimpleNamespace(last_modified=modified, files=[file]))
        return str(link)

    def scan(self):
        return SimpleNamespace(repos=self.repos)


class LiteRTModelsTestCase(unittest.TestCase):
    def setUp(self):
        self.cache = Cache()
        self.addCleanup(self.cache.folder.cleanup)
        # The models folder, which doesn't exist until something's put in it
        local = tempfile.TemporaryDirectory()
        self.addCleanup(local.cleanup)
        self.local = os.path.join(local.name, "models")
        self.chat_files = set()
        for patch in (mock.patch("huggingface_hub.scan_cache_dir", self.cache.scan),
                      mock.patch.object(litert_models, "facts",
                                        lambda path: TEXT if os.path.basename(path) in self.chat_files else None),
                      mock.patch.object(litert_models, "total_memory", lambda: 8 * GB),
                      mock.patch.object(litert_models, "local_folder", lambda: self.local)):
            patch.start()
            self.addCleanup(patch.stop)

    def download(self, repo_id, name, chat=True, **options):
        if chat:
            self.chat_files.add(name)
        return self.cache.add(repo_id, name, **options)

    def hub(self, *repos):
        self.fake_hub = FakeHub(*repos)
        patch = mock.patch("huggingface_hub.HfApi", lambda: self.fake_hub)
        patch.start()
        self.addCleanup(patch.stop)


class TestDownloaded(LiteRTModelsTestCase):
    def test_chat_models_by_id(self):
        qwen = self.download("litert-community/Qwen3-0.6B", "Qwen3-0.6B.litertlm")
        gemma = self.download("litert-community/gemma-4-E2B-it-litert-lm", "gemma-4-E2B-it.litertlm")
        self.download("litert-community/embeddinggemma", "embeddinggemma.litertlm", chat=False)
        self.download("someone/other", "README.md")
        self.assertEqual(litert_models.downloaded(), {"gemma-4-E2B-it": gemma, "Qwen3-0.6B": qwen})

    def test_newest_revision_wins(self):
        self.download("a/model", "model.litertlm", modified=1)
        newer = self.download("a/model", "model.litertlm", modified=2)
        self.assertEqual(litert_models.downloaded(), {"model": newer})


class TestPickFile(unittest.TestCase):
    def test_the_file_named_after_the_repo(self):
        files = {"Qwen3-0.6B.litertlm": 6, "qwen3_0.6b_q4.litertlm": 3}
        self.assertEqual(litert_models.pick_file("litert-community/Qwen3-0.6B", files), "Qwen3-0.6B.litertlm")
        self.assertEqual(litert_models.pick_file("litert-community/gemma-4-E2B-it-litert-lm",
                                                 {"gemma-4-E2B-it.litertlm": 2, "gemma-4-E2B-it-web.litertlm": 1}),
                         "gemma-4-E2B-it.litertlm")
        self.assertEqual(litert_models.pick_file("litert-community/Qwen3-1.7B",
                                                 {"Qwen3-1.7B_dynamic_wi4b32_afp32.litertlm": 1, "Qwen3_1.7B.litertlm": 2}),
                         "Qwen3_1.7B.litertlm")

    def test_else_the_smallest_for_any_device(self):
        files = {"model_q8.litertlm": 8, "model_q4.mediatek.mt6993.litertlm": 2, "model_q4-web.litertlm": 2,
                 "model_q4_gpu.litertlm": 2, "model_q4.qualcomm.sm8750.litertlm": 2, "model_q6.litertlm": 6,
                 "sub/model_q1.litertlm": 1, "model.tflite": 1}
        self.assertEqual(litert_models.pick_file("a/model", files), "model_q6.litertlm")

    def test_none(self):
        self.assertIsNone(litert_models.pick_file("a/model", {"model.tflite": 1}))


class TestReference(unittest.TestCase):
    def test_addresses_and_names(self):
        for text, expected in [
                ("litert-community/Qwen3-0.6B", ("litert-community/Qwen3-0.6B", None)),
                (" https://huggingface.co/litert-community/Qwen3-0.6B/ ", ("litert-community/Qwen3-0.6B", None)),
                ("https://huggingface.co/a/b/blob/main/x.litertlm", ("a/b", "x.litertlm")),
                ("https://huggingface.co/a/b/resolve/abc123/sub/x.litertlm?download=true", ("a/b", "sub/x.litertlm")),
                ("hf.co/a/b/tree/main", ("a/b", None)),
                ("a/b/x.litertlm", ("a/b", "x.litertlm"))]:
            self.assertEqual(litert_models.reference(text), expected, text)
        for text in ("qwen3", "https://example.com/a/b", "a b/c"):
            with self.assertRaises(ValueError, msg=text):
                litert_models.reference(text)


class TestLocalFolder(LiteRTModelsTestCase):
    def put(self, name, chat=True):
        os.makedirs(self.local, exist_ok=True)
        if chat:
            self.chat_files.add(name)
        path = os.path.join(self.local, name)
        with open(path, "wb") as f:
            f.write(b"x" * 300)
        return path

    def test_listed_and_preferred(self):
        self.download("a/model", "model.litertlm")
        mine = self.put("model.litertlm")
        self.put("notes.litertlm", chat=False)
        self.assertEqual(litert_models.downloaded(), {"model": mine})

    def test_add_a_file_and_remove_it(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            path = os.path.join(elsewhere, "mine.litertlm")
            with open(path, "wb") as f:
                f.write(b"x" * 300)
            with self.assertRaisesRegex(ValueError, "isn't a chat model"):
                litert_models.add_file(path)
            self.chat_files.add("mine.litertlm")
            self.assertEqual(litert_models.add_file(path), "mine")
        self.assertIn("mine", litert_models.downloaded())
        self.assertEqual(litert_models.remove("mine"), 300)
        self.assertEqual(litert_models.downloaded(), {})
        with self.assertRaisesRegex(ValueError, "isn't a .litertlm file"):
            litert_models.add_file("/nope/model.bin")


class TestFits(unittest.TestCase):
    def test_leaves_room_for_the_rest_of_the_server(self):
        with mock.patch.object(litert_models, "total_memory", lambda: 8 * 2**30):
            self.assertTrue(litert_models.fits(5 * 2**30))
            self.assertFalse(litert_models.fits(5 * 2**30 + 1))
            self.assertIsNone(litert_models.fits(None))
        with mock.patch.object(litert_models, "total_memory", lambda: None):
            self.assertIsNone(litert_models.fits(1))


class TestSearch(LiteRTModelsTestCase):
    def test_describes_chat_models(self):
        self.download("litert-community/Qwen3-0.6B", "Qwen3-0.6B.litertlm")
        self.hub(hub_repo("litert-community/Qwen3-0.6B", {"Qwen3-0.6B.litertlm": GB, "q4.litertlm": GB // 2},
                          downloads=50),
                 hub_repo("litert-community/Qwen3-14B", {"qwen3_14b.litertlm": 9 * GB}),
                 hub_repo("litert-community/whisper", {"whisper.litertlm": GB}, tags=["automatic-speech-recognition"]),
                 hub_repo("litert-community/embeddinggemma-300m", {"embeddinggemma.litertlm": GB}),
                 hub_repo("litert-community/Qwen3-TTS-12Hz-0.6B-Base", {"qwen3_tts_int4.litertlm": GB}),
                 hub_repo("warped-community/Qwen3-ASR-0.6B-litert-lm", {"qwen3_asr_0.6b.litertlm": GB}),
                 hub_repo("someone/Qwen3-0.6B-SM8650-LiteRT-LM", {"model.litertlm": GB}),
                 hub_repo("litert-community/Qwen3-0.6B-Base", {"Qwen3-0.6B-Base.litertlm": GB}),
                 hub_repo("litert-community/gemma-3-1b-pt-litert-lm", {"gemma-3-1b-pt.litertlm": GB}),
                 hub_repo("NeuML/gemma-4-tiny-random-litert-lm", {"gemma-4-tiny-random.litertlm": GB}),
                 hub_repo("litert-community/no-files", {"model.tflite": GB}))
        found = litert_models.search()
        self.assertEqual([m["repo"] for m in found], ["litert-community/Qwen3-0.6B", "litert-community/Qwen3-14B"])
        self.assertEqual(found[0], {"repo": "litert-community/Qwen3-0.6B", "file": "Qwen3-0.6B.litertlm",
                                    "id": "Qwen3-0.6B", "size": GB, "fits": True, "downloaded": True,
                                    "downloads": 50, "likes": 0, "other_files": ["q4.litertlm"],
                                    "gated": False})
        self.assertEqual((found[1]["fits"], found[1]["downloaded"]), (False, False))

    def test_query(self):
        self.hub(hub_repo("a/Qwen3", {"Qwen3.litertlm": 1}), hub_repo("a/gemma", {"gemma.litertlm": 1}))
        self.assertEqual([m["id"] for m in litert_models.search("qwen")], ["Qwen3"])

    def test_sort(self):
        self.hub(hub_repo("a/Qwen3", {"Qwen3.litertlm": 1}))
        litert_models.search(sort="trending")
        self.assertEqual(self.fake_hub.sort, "trending_score")
        with self.assertRaises(ValueError):
            litert_models.search(sort="best")

    def test_a_named_repo_alone_whatever_its_name(self):
        self.hub(hub_repo("a/Qwen3-0.6B-Base", {"base.litertlm": 1, "base_q4.litertlm": 2}),
                 hub_repo("a/Qwen3", {"Qwen3.litertlm": 1}))
        found = litert_models.search("https://huggingface.co/a/Qwen3-0.6B-Base/blob/main/base_q4.litertlm")
        self.assertEqual([(m["repo"], m["file"]) for m in found], [("a/Qwen3-0.6B-Base", "base_q4.litertlm")])
        self.assertEqual(litert_models.search("a/Qwen3-0.6B-Base")[0]["file"], "base.litertlm")

    def test_gated_and_missing_repos_say_what_to_do(self):
        self.hub()
        with self.assertRaisesRegex(ValueError, "gated: accept its terms .*hf auth login"):
            litert_models.search("gated/model")
        with self.assertRaisesRegex(ValueError, "There's no repo a/missing"):
            litert_models.search("a/missing")
        with self.assertRaisesRegex(ValueError, "has no LiteRT-LM file"):
            self.hub(hub_repo("a/b", {"model.tflite": 1}))
            litert_models.search("a/b")


class TestAdd(LiteRTModelsTestCase):
    def setUp(self):
        super().setUp()
        self.hub(hub_repo("litert-community/Qwen3-0.6B", {"Qwen3-0.6B.litertlm": 10, "q4.litertlm": 5}))
        patch = mock.patch("huggingface_hub.hf_hub_download", self.fake_download)
        patch.start()
        self.addCleanup(patch.stop)

    def fake_download(self, repo_id, name, tqdm_class=None):
        return self.download(repo_id, name)

    def test_picked_file(self):
        self.assertEqual(litert_models.add("litert-community/Qwen3-0.6B"), "Qwen3-0.6B")
        self.assertIn("Qwen3-0.6B", litert_models.downloaded())

    def test_named_file(self):
        self.assertEqual(litert_models.add("litert-community/Qwen3-0.6B", "q4.litertlm"), "q4")

    def test_an_address(self):
        self.assertEqual(litert_models.add("https://huggingface.co/litert-community/Qwen3-0.6B/blob/main/q4.litertlm"),
                         "q4")

    def test_unknown_file(self):
        with self.assertRaisesRegex(ValueError, "no LiteRT-LM file named nope.litertlm"):
            litert_models.add("litert-community/Qwen3-0.6B", "nope.litertlm")

    def test_a_gated_download_says_what_to_do(self):
        from huggingface_hub.errors import GatedRepoError

        def gated(repo_id, name, tqdm_class=None):
            raise GatedRepoError("gated", response=mock.Mock(status_code=401))
        with mock.patch("huggingface_hub.hf_hub_download", gated):
            with self.assertRaisesRegex(ValueError, "is gated: accept its terms"):
                litert_models.add("litert-community/Qwen3-0.6B")

    def test_not_a_chat_model(self):
        self.fake_download = lambda repo_id, name, tqdm_class=None: self.download(repo_id, name, chat=False)
        with mock.patch("huggingface_hub.hf_hub_download", self.fake_download):
            with self.assertRaisesRegex(ValueError, "isn't a chat model"):
                litert_models.add("litert-community/Qwen3-0.6B")


class TestRemove(LiteRTModelsTestCase):
    def test_deletes_the_file(self):
        link = self.download("litert-community/Qwen3-0.6B", "Qwen3-0.6B.litertlm", size=600)
        self.assertEqual(litert_models.remove("Qwen3-0.6B"), 600)
        self.assertFalse(os.path.lexists(link))
        self.assertFalse((Path(self.cache.folder.name) / "litert-community--Qwen3-0.6B").exists())

    def test_keeps_the_repos_other_files(self):
        self.download("litert-community/Qwen3-0.6B", "Qwen3-0.6B.litertlm")
        other = self.download("litert-community/Qwen3-0.6B", "q4.litertlm")
        litert_models.remove("Qwen3-0.6B")
        self.assertTrue(os.path.exists(other))

    def test_not_downloaded(self):
        with self.assertRaises(LookupError):
            litert_models.remove("Qwen3-0.6B")


class TestCommand(LiteRTModelsTestCase):
    def run_command(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = cli.main(["models", *argv])
        return status, out.getvalue()

    def test_list(self):
        self.assertIn("No LiteRT-LM models are downloaded", self.run_command()[1])
        self.download("litert-community/Qwen3-0.6B", "Qwen3-0.6B.litertlm", size=600_000_000)
        status, out = self.run_command("list")
        self.assertEqual(status, 0)
        self.assertEqual(out.splitlines()[1].split(), ["Qwen3-0.6B", "0.6", "GB", "text"])

    def test_search(self):
        self.hub(hub_repo("litert-community/Qwen3-14B", {"qwen3_14b.litertlm": 9 * GB}, downloads=7))
        status, out = self.run_command("search", "qwen3")
        self.assertEqual(status, 0)
        self.assertEqual(out.splitlines()[1].split(),
                         ["litert-community/Qwen3-14B", "qwen3_14b.litertlm", "9.0", "GB", "too", "big", "7"])

    def test_add_and_remove(self):
        self.hub(hub_repo("litert-community/Qwen3-0.6B", {"Qwen3-0.6B.litertlm": 10}))
        with mock.patch("huggingface_hub.hf_hub_download",
                        lambda repo_id, name, tqdm_class=None: self.download(repo_id, name, size=2 * GB)):
            status, out = self.run_command("add", "litert-community/Qwen3-0.6B")
        self.assertEqual(status, 0)
        self.assertIn("Added Qwen3-0.6B", out)
        self.assertEqual(self.run_command("remove", "Qwen3-0.6B"), (0, "Removed Qwen3-0.6B, freeing 2.0 GB.\n"))
        self.assertEqual(self.run_command("remove", "Qwen3-0.6B"), (1, "Qwen3-0.6B isn't downloaded.\n"))

    def test_add_a_local_file(self):
        with tempfile.TemporaryDirectory() as elsewhere:
            path = os.path.join(elsewhere, "mine.litertlm")
            with open(path, "wb") as f:
                f.write(b"x")
            self.chat_files.add("mine.litertlm")
            status, out = self.run_command("add", path)
        self.assertEqual(status, 0)
        self.assertIn("Added mine", out)
        self.assertTrue(os.path.isfile(os.path.join(self.local, "mine.litertlm")))


if __name__ == "__main__":
    unittest.main()
