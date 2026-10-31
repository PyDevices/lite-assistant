"""
load_skill: instructions the model loads when a task needs them, as the
Google AI Edge Gallery's Agent Skills and Claude's skills work. Each skill
is a folder with a SKILL.md, whose frontmatter gives its name and
description. Only those go to the model, in the tool's description, so a
dozen skills cost a few lines of context until one is used.

The system's skills are in the skills folder of the data directory, and
each user's own in their folder's; a user's skill takes the place of a
system skill with the same name.
"""
import os
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

from .. import paths

# The longest description listed, in characters, and the most skills listed
MAX_DESCRIPTION, MAX_SKILLS = 200, 40
# The most characters of a skill's instructions or files returned
MAX_TEXT = 12000


def frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    """
    A SKILL.md's frontmatter fields, simple YAML "key: value" lines with
    indented lines continuing a value, and the text after it.
    """
    found = re.match(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", text, re.S)
    if not found:
        return {}, text
    fields: Dict[str, str] = {}
    key = None
    for line in found.group(1).splitlines():
        pair = re.match(r"^([\w-]+):\s*(.*)$", line)
        if pair:
            key = pair.group(1)
            fields[key] = pair.group(2).strip()
        elif key and line.startswith((" ", "\t")):
            fields[key] = (fields[key] + " " + line.strip()).strip()
    # A YAML block marker or quotes aren't part of the value
    return {k: re.sub(r"^[>|][-+]?\s*", "", v).strip().strip("\"'") for k, v in fields.items()}, text[found.end():]


class Skills:
    def __init__(self, folders: Union[str, Sequence[str]] = ""):
        # Where later folders have a skill of the same name, theirs is used
        folders = [folders] if isinstance(folders, str) else list(folders)
        self.folders = [folder or paths.data_dir("skills") for folder in folders] or [paths.data_dir("skills")]
        # name -> (folder, description), and the folder's state when they were read
        self.skills: Dict[str, Tuple[str, str]] = {}
        self.seen: Optional[Tuple] = None
        self.load_skill = self.tool()
        self.refresh()

    def state(self) -> Tuple:
        """ The SKILL.md files and when each changed, to tell when to read them again. """
        found = []
        for folder in self.folders:
            if os.path.isdir(folder):
                for name in sorted(os.listdir(folder)):
                    path = os.path.join(folder, name, "SKILL.md")
                    if os.path.isfile(path):
                        found.append((path, os.path.getmtime(path)))
        return tuple(found)

    def refresh(self) -> None:
        """ Reads the skills again if they've changed, and lists them in load_skill's description. """
        state = self.state()
        if state == self.seen:
            return
        self.seen = state
        self.skills = {}
        for path, _ in state:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    fields, _ = frontmatter(f.read())
            except OSError:
                continue
            name = fields.get("name") or os.path.basename(os.path.dirname(path))
            description = " ".join(fields.get("description", "").split())
            if len(description) > MAX_DESCRIPTION:
                description = description[:MAX_DESCRIPTION - 3].rstrip() + "..."
            self.skills.pop(name, None)
            self.skills[name] = (os.path.dirname(path), description)
        self.skills = dict(list(self.skills.items())[-MAX_SKILLS:])
        listed = "\n".join(f"- {name}: {description}" for name, (_, description) in self.skills.items())
        self.load_skill.__doc__ = (
            "Loads a skill: instructions for a kind of task. When the user asks for something one of these "
            "skills is for, load it first and follow its instructions.\n"
            f"{listed or '(There are no skills yet.)'}\n\n"
            "Args:\n"
            "    name: The skill's name.\n"
            "    file: A file the skill's instructions name, to read it; leave it out for the instructions.")
        if self.skills:
            self.load_skill.__annotations__["name"] = Literal[tuple(self.skills)]

    def tool(self) -> Callable:
        def load_skill(name: str, file: Optional[str] = None) -> Dict[str, Any]:
            self.refresh()
            if name not in self.skills:
                return {"error": f"There's no skill named {name!r}. The skills are: {', '.join(self.skills) or 'none'}."}
            folder = os.path.realpath(self.skills[name][0])
            path = os.path.realpath(os.path.join(folder, file or "SKILL.md"))
            # A skill reads only its own files
            if os.path.commonpath([folder, path]) != folder or not os.path.isfile(path):
                return {"error": f"The skill {name} has no file {file!r}."}
            with open(path, encoding="utf-8", errors="replace") as f:
                text = f.read()
            if not file:
                text = frontmatter(text)[1]
            found: Dict[str, Any] = {"skill": name, "instructions" if not file else "text": text.strip()[:MAX_TEXT]}
            if not file:
                # Small models otherwise treat the instructions as news, and ask what to do
                found["note"] = "Follow these instructions to do what the user asked, now."
                others = sorted(os.path.relpath(os.path.join(root, n), folder) for root, _, names in os.walk(folder)
                                for n in names if n != "SKILL.md")
                if others:
                    found["files"] = others[:50]
            return found
        load_skill.refresh = self.refresh
        # Small models call a skill by its own name, as if it were a tool; such a call loads it
        load_skill.answers = lambda name: name in self.skills and {"name": name}
        return load_skill

    def names(self) -> List[str]:
        return list(self.skills)


def user_skills(system: str = "") -> Callable:
    """
    load_skill for the system's skills, whose for_user gives the tool for a
    user, with their own skills too, or None if there are none to offer.
    """
    every: Dict[Tuple[str, ...], Skills] = {}

    def for_user(user) -> Optional[Callable]:
        folders = (system or paths.data_dir("skills"),) + ((user.folder("skills"),) if user else ())
        if folders not in every:
            every[folders] = Skills(folders)
        skills = every[folders]
        skills.refresh()
        return skills.load_skill if skills.skills else None

    tool = Skills(system).load_skill
    tool.for_user = for_user
    return tool
