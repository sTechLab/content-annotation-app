import tempfile
import unittest
from pathlib import Path

import pandas as pd

from app import annotation_app
from scripts import merge_annotations


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

    def test_configured_options_are_available_only_in_full_mode(self) -> None:
        configured = ["Recommendation"]
        coder_options = ["Criticism"]

        self.assertEqual(
            annotation_app.available_coding_options(
                configured,
                coder_options,
                annotation_app.SAMPLE_MODE,
            ),
            ["Criticism"],
        )
        self.assertEqual(
            annotation_app.available_coding_options(
                configured,
                coder_options,
                annotation_app.FULL_MODE,
            ),
            ["Recommendation", "Criticism"],
        )

    def test_sample_order_is_fixed_and_full_order_is_stable_per_coder(self) -> None:
        posts = pd.DataFrame({"item_id": [str(value) for value in range(20)]})
        original_order = posts["item_id"].tolist()

        sample = annotation_app.order_posts_for_mode(
            posts,
            annotation_app.SAMPLE_MODE,
            "Nadav",
            "2026-06-26",
        )
        full_nadav = annotation_app.order_posts_for_mode(
            posts,
            annotation_app.FULL_MODE,
            "Nadav",
            "2026-06-26",
        )
        full_nadav_again = annotation_app.order_posts_for_mode(
            posts,
            annotation_app.FULL_MODE,
            "Nadav",
            "2026-06-26",
        )
        full_teammate = annotation_app.order_posts_for_mode(
            posts,
            annotation_app.FULL_MODE,
            "Teammate",
            "2026-06-26",
        )

        self.assertEqual(sample["item_id"].tolist(), original_order)
        self.assertEqual(
            full_nadav["item_id"].tolist(),
            full_nadav_again["item_id"].tolist(),
        )
        self.assertNotEqual(full_nadav["item_id"].tolist(), original_order)
        self.assertNotEqual(
            full_nadav["item_id"].tolist(),
            full_teammate["item_id"].tolist(),
        )

    def test_sample_and_full_use_separate_annotation_files(self) -> None:
        root = Path("annotations")

        sample_path = annotation_app.annotation_file_path(
            root,
            "Nadav",
            annotation_app.SAMPLE_MODE,
        )
        full_path = annotation_app.annotation_file_path(
            root,
            "Nadav",
            annotation_app.FULL_MODE,
        )

        self.assertEqual(sample_path.name, "Nadav_annotations.csv")
        self.assertEqual(full_path.name, "Nadav_full_annotations.csv")

    def test_annotation_and_merge_columns_stay_aligned(self) -> None:
        self.assertEqual(
            annotation_app.ANNOTATION_COLUMNS,
            merge_annotations.ANNOTATION_COLUMNS,
        )

    def test_modes_have_distinct_ids_and_survive_merge(self) -> None:
        post = self.posts.iloc[0]
        rows = []
        for mode in [annotation_app.SAMPLE_MODE, annotation_app.FULL_MODE]:
            rows.append(
                annotation_app.build_annotation_row(
                    post=post,
                    coder_name="Nadav",
                    annotation_mode=mode,
                    skipped=False,
                    is_relevant="Yes",
                    tags_selected=[],
                    tags_added=[],
                    labels_selected=[],
                    labels_added=[],
                    starred=False,
                    coder_notes="",
                )
            )

        self.assertNotEqual(rows[0]["annotation_id"], rows[1]["annotation_id"])

        with tempfile.TemporaryDirectory() as temporary_directory:
            raw_dir = Path(temporary_directory)
            pd.DataFrame([rows[0]]).to_csv(
                raw_dir / "nadav_annotations.csv",
                index=False,
            )
            pd.DataFrame([rows[1]]).to_csv(
                raw_dir / "nadav_full_annotations.csv",
                index=False,
            )
            merged = merge_annotations.merge_annotations(raw_dir)

        self.assertEqual(len(merged), 2)
        self.assertEqual(
            set(merged["annotation_mode"]),
            {annotation_app.SAMPLE_MODE, annotation_app.FULL_MODE},
        )

    def test_progress_includes_annotated_and_skipped_posts(self) -> None:
        annotated = annotation_app.build_annotation_row(
            post=self.posts.iloc[0],
            coder_name="Nadav",
            annotation_mode=annotation_app.SAMPLE_MODE,
            skipped=False,
            is_relevant="Yes",
            tags_selected=[],
            tags_added=[],
            labels_selected=[],
            labels_added=[],
            starred=False,
            coder_notes="",
        )
        skipped = annotation_app.build_annotation_row(
            post=self.posts.iloc[1],
            coder_name="Nadav",
            annotation_mode=annotation_app.SAMPLE_MODE,
            skipped=True,
            is_relevant="",
            tags_selected=[],
            tags_added=[],
            labels_selected=[],
            labels_added=[],
            starred=False,
            coder_notes="",
        )
        annotations = pd.DataFrame([annotated, skipped])

        counts = annotation_app.progress_counts(self.posts, annotations, "Nadav")

        self.assertEqual(counts, (1, 1, 2))
        self.assertEqual(annotation_app.handled_progress(1, 1, 4), 0.5)

    def test_saving_changes_replaces_the_existing_row(self) -> None:
        post = self.posts.iloc[0]
        first = annotation_app.build_annotation_row(
            post=post,
            coder_name="Nadav",
            annotation_mode=annotation_app.SAMPLE_MODE,
            skipped=False,
            is_relevant="Yes",
            tags_selected=["Recommendation"],
            tags_added=[],
            labels_selected=[],
            labels_added=[],
            starred=False,
            coder_notes="First answer",
        )
        updated = annotation_app.build_annotation_row(
            post=post,
            coder_name="Nadav",
            annotation_mode=annotation_app.SAMPLE_MODE,
            skipped=False,
            is_relevant="No",
            tags_selected=["Criticism"],
            tags_added=[],
            labels_selected=["Posts a lot"],
            labels_added=[],
            starred=True,
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
        self.assertEqual(saved.iloc[0]["star"], "True")
        self.assertEqual(saved.iloc[0]["coder_notes"], "Updated answer")

    def test_marking_an_answer_skipped_clears_coding_fields(self) -> None:
        row = annotation_app.build_annotation_row(
            post=self.posts.iloc[0],
            coder_name="Nadav",
            annotation_mode=annotation_app.FULL_MODE,
            skipped=True,
            is_relevant="Yes",
            tags_selected=["Recommendation"],
            tags_added=["Positive"],
            labels_selected=["Posts a lot"],
            labels_added=["Replies a lot"],
            starred=True,
            coder_notes="Could not determine the context",
        )

        self.assertTrue(row["skipped"])
        self.assertTrue(row["skip_datetime"])
        self.assertEqual(row["is_relevant"], "")
        self.assertEqual(row["tags_final"], "")
        self.assertEqual(row["labels_final"], "")
        self.assertTrue(row["star"])
        self.assertEqual(row["coder_notes"], "Could not determine the context")

    def test_old_annotation_csv_gets_sample_mode_and_blank_star(self) -> None:
        old_columns = [
            column
            for column in annotation_app.ANNOTATION_COLUMNS
            if column not in {"annotation_mode", "star"}
        ]
        old_columns.insert(3, "clean_coding_mode")
        old_row = {column: "" for column in old_columns}
        old_row["clean_coding_mode"] = "False"
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "old_annotations.csv"
            pd.DataFrame([old_row], columns=old_columns).to_csv(path, index=False)
            loaded = annotation_app.load_annotations(path)

        self.assertIn("star", loaded.columns)
        self.assertEqual(list(loaded.columns), annotation_app.ANNOTATION_COLUMNS)
        self.assertEqual(loaded.iloc[0]["annotation_mode"], annotation_app.SAMPLE_MODE)
        self.assertEqual(loaded.iloc[0]["star"], "")


if __name__ == "__main__":
    unittest.main()
