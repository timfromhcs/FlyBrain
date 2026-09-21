"""V6 prompt templates: versioned, complete, renderable, validated."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import unittest

from src.prompts.render import load_templates, render, extract_json, check_output


class TestTemplates(unittest.TestCase):
    def test_all_templates_complete(self):
        t = load_templates()
        self.assertTrue(t["version"].startswith("prompts_v"))
        for name in ("goal_decision", "dream_scenario", "visual_concept",
                     "conversation_reply"):
            self.assertIn(name, t)
            for field in ("system", "user", "schema", "temperature", "max_tokens"):
                self.assertIn(field, t[name], f"{name} missing {field}")

    def test_render_fills_and_validates(self):
        t = load_templates()
        r = render("goal_decision", t, world="meadow", needs="hunger 0.8",
                   memories="[]")
        self.assertIn("meadow", r["user"])
        self.assertIn("{", r["user"])  # schema braces survive formatting
        with self.assertRaises(ValueError):
            render("goal_decision", t, world="x")
        with self.assertRaises(KeyError):
            render("nope", t)

    def test_extract_json_think_trace(self):
        text = '<think>\nLet me think {"a": 1} hmm.\n</think>\n{"goal": "explore", "urgency": 0.5}'
        d = extract_json(text, "goal")
        self.assertEqual(check_output(d, ["goal", "urgency"])["goal"], "explore")
        with self.assertRaises(ValueError):
            extract_json("no json here", "goal")
        with self.assertRaises(ValueError):
            check_output({"goal": "x"}, ["goal", "urgency"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
