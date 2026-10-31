"""
Skills: reading SKILL.md frontmatter, load_skill's description listing
the skills and refreshing as they change, and loading one's instructions
and files, but no file outside it.
"""
import json
import os
import tempfile
import time
import unittest
from unittest import mock

from lite_assistant.engines import ToolCall
from lite_assistant.tools import definition, stand_in
from lite_assistant.tools.skills import Skills, frontmatter, user_skills
from lite_assistant.users import User

SCALER = """---
name: recipe-scaler
description: >
  Scales a recipe's quantities
  to a number of servings.
---
# Scaling a recipe

Multiply each quantity by the servings wanted over the servings made. See units.md.
"""


class TestSkills(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def add(self, folder, text, **files):
        os.makedirs(os.path.join(self.folder.name, folder), exist_ok=True)
        for name, content in {"SKILL.md": text, **files}.items():
            with open(os.path.join(self.folder.name, folder, name), "w") as f:
                f.write(content)

    def test_frontmatter(self):
        fields, body = frontmatter(SCALER)
        self.assertEqual(fields, {"name": "recipe-scaler",
                                  "description": "Scales a recipe's quantities to a number of servings."})
        self.assertTrue(body.startswith("# Scaling a recipe"))
        self.assertEqual(frontmatter("No frontmatter."), ({}, "No frontmatter."))

    def test_the_description_lists_the_skills(self):
        self.add("scaler", SCALER)
        self.add("plain", "Just instructions.")
        tool = Skills(self.folder.name).load_skill
        described = definition(tool)["function"]
        self.assertIn("- recipe-scaler: Scales a recipe's quantities to a number of servings.", described["description"])
        # A skill without frontmatter goes by its folder's name
        self.assertIn("- plain:", described["description"])
        self.assertEqual(described["parameters"]["properties"]["name"]["enum"], ["plain", "recipe-scaler"])
        self.assertEqual(described["parameters"]["required"], ["name"])

    def test_a_new_skill_is_listed(self):
        skills = Skills(self.folder.name)
        self.assertIn("There are no skills yet", definition(skills.load_skill)["function"]["description"])
        self.add("scaler", SCALER)
        self.assertIn("recipe-scaler", definition(skills.load_skill)["function"]["description"])
        # A changed one is read again
        time.sleep(0.01)
        self.add("scaler", SCALER.replace("Scales", "Resizes"))
        os.utime(os.path.join(self.folder.name, "scaler", "SKILL.md"), (time.time() + 5, time.time() + 5))
        self.assertIn("Resizes", definition(skills.load_skill)["function"]["description"])

    def test_loading_a_skill_and_its_files(self):
        self.add("scaler", SCALER, **{"units.md": "A cup is 240 ml."})
        load = Skills(self.folder.name).load_skill
        loaded = load("recipe-scaler")
        self.assertTrue(loaded["instructions"].startswith("# Scaling a recipe"))
        self.assertNotIn("description:", loaded["instructions"])
        self.assertEqual(loaded["files"], ["units.md"])
        self.assertEqual(load("recipe-scaler", "units.md"), {"skill": "recipe-scaler", "text": "A cup is 240 ml."})
        self.assertIn("error", load("nope"))
        self.assertIn("error", load("recipe-scaler", "../scaler/../../secret.txt"))
        self.assertIn("error", load("recipe-scaler", "missing.md"))

    def test_a_call_to_a_skill_by_name_loads_it(self):
        self.add("scaler", SCALER)
        load = Skills(self.folder.name).load_skill
        call, function = stand_in(ToolCall("recipe-scaler", '{"servings": 4}', "call_1"), {"load_skill": load})
        self.assertIs(function, load)
        self.assertEqual((call.name, json.loads(call.arguments), call.id), ("load_skill", {"name": "recipe-scaler"}, "call_1"))
        self.assertIsNone(stand_in(ToolCall("nope", "{}"), {"load_skill": load})[1])


    def test_each_user_has_the_systems_skills_and_their_own(self):
        with mock.patch.dict(os.environ, {"LITE_ASSISTANT_HOME": self.folder.name}):
            tool = user_skills(os.path.join(self.folder.name, "system"))
            alice, bob = User("alice"), User("bob")
            # No skills anywhere: load_skill isn't offered
            self.assertIsNone(tool.for_user(alice))
            self.add("system/scaler", SCALER)
            self.add("system/plain", "The system's plain skill.")
            self.add("users/alice/skills/plain", "Alice's plain skill.")
            self.add("users/alice/skills/hers", "Only Alice's.")
            alices, bobs = tool.for_user(alice), tool.for_user(bob)
            self.assertEqual(alices("plain")["instructions"], "Alice's plain skill.")
            self.assertEqual(bobs("plain")["instructions"], "The system's plain skill.")
            self.assertIn("error", bobs("hers"))
            self.assertCountEqual(definition(alices)["function"]["parameters"]["properties"]["name"]["enum"],
                                  ["recipe-scaler", "plain", "hers"])
            self.assertEqual(definition(tool.for_user(None))["function"]["parameters"]["properties"]["name"]["enum"],
                             ["plain", "recipe-scaler"])


if __name__ == "__main__":
    unittest.main()
