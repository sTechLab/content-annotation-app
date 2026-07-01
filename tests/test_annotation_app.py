import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app import annotation_app


class AnnotationNavigationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.posts = pd.DataFrame(
            [
                {"item_id": "1", "post_url": "https://example.com/1"},
                {"item_id": "2", "post_url": "https://example.com/2"},
                {"item_id": "3", "post_url": "https://example.com/3"},
                {"item_id": "4", "post_url": "https://example.com/4"},
            ]
        )

    def test_previous_handled_item_follows_post_order(self) -> None:
        handled = {"1", "2", "3"}

        self.assertEqual(
            annotation_app.previous_handled_item_id(self.posts, handled, "4"),
            "3",
        )
        self.assertEqual(
            annotation_app.previous_handled_item_id(self.posts, handled, "3"),
            "2",
        )
        self.assertEqual(
            annotation_app.previous_handled_item_id(self.posts, handled, "2"),
            "1",
        )
        self.assertIsNone(
            annotation_app.previous_handled_item_id(self.posts, handled, "1")
        )
        self.assertEqual(
            annotation_app.previous_handled_item_id(self.posts, handled),
            "3",
        )

    def test_next_item_follows_post_order(self) -> None:
        self.assertEqual(annotation_app.next_item_id(self.posts, "1"), "2")
        self.assertEqual(annotation_app.next_item_id(self.posts, "2"), "3")
        self.assertIsNone(annotation_app.next_item_id(self.posts, "4"))
        self.assertIsNone(annotation_app.next_item_id(self.posts, "missing"))

    def test_saving_changes_replaces_the_existing_row(self) -> None:
        post = self.posts.iloc[0]
        first = annotation_app.build_annotation_row(
            post=post,
            coder_name="Nadav",
            clean_coding_mode=True,
            skipped=False,
            is_relevant="Yes",
            tags_selected=["Recommendation"],
            tags_added=[],
            labels_selected=[],
            labels_added=[],
            coder_notes="First answer",
        )
        updated = annotation_app.build_annotation_row(
            post=post,
            coder_name="Nadav",
            clean_coding_mode=True,
            skipped=False,
            is_relevant="No",
            tags_selected=["Criticism"],
            tags_added=[],
            labels_selected=["Posts a lot"],
            labels_added=[],
            coder_notes="Updated answer",
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "annotations.csv"
            annotation_app.save_annotation(path, first)
            annotation_app.save_annotation(path, updated)
            saved = annotation_app.load_annotations(path)

        self.assertEqual(len(saved), 1)
        self.assertEqual(saved.iloc[0]["annotation_id"], first["annotation_id"])
        self.assertEqual(saved.iloc[0]["is_relevant"], "No")
        self.assertEqual(saved.iloc[0]["tags_final"], "Criticism")
        self.assertEqual(saved.iloc[0]["labels_final"], "Posts a lot")
        self.assertEqual(saved.iloc[0]["coder_notes"], "Updated answer")

    def test_marking_an_answer_skipped_clears_coding_fields(self) -> None:
        row = annotation_app.build_annotation_row(
            post=self.posts.iloc[0],
            coder_name="Nadav",
            clean_coding_mode=True,
            skipped=True,
            is_relevant="Yes",
            tags_selected=["Recommendation"],
            tags_added=["Positive"],
            labels_selected=["Posts a lot"],
            labels_added=["Replies a lot"],
            coder_notes="Could not determine the context",
        )

        self.assertTrue(row["skipped"])
        self.assertTrue(row["skip_datetime"])
        self.assertEqual(row["is_relevant"], "")
        self.assertEqual(row["tags_final"], "")
        self.assertEqual(row["labels_final"], "")
        self.assertEqual(row["coder_notes"], "Could not determine the context")


if __name__ == "__main__":
    unittest.main()
