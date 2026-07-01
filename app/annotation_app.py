"""Local Streamlit app for annotating collected Bluesky posts."""

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.json"
TAGS_PATH = PROJECT_ROOT / "config" / "tags.json"
PUBLIC_BSKY_THREAD_API = "https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread"
THREAD_DEPTH = 100
THREAD_PARENT_HEIGHT = 100

DEFAULT_CONFIG = {
    "processed_data_dir": "data/processed",
    "processed_data_filename": "enriched_results_sample.csv",
    "annotations_dir": "annotations",
    "collection_date": "",
}

POST_COLUMNS = [
    "item_id",
    "post_url",
    "post_uri",
    "post_cid",
    "author_handle",
    "author_did",
    "post_datetime",
    "text",
    "post_type",
    "parent_post_uri",
    "root_post_uri",
    "language",
    "like_count",
    "repost_count",
    "reply_count",
    "quote_count",
]

ANNOTATION_COLUMNS = [
    "annotation_id",
    "annotation_datetime",
    "coder_name",
    "clean_coding_mode",
    "item_id",
    "post_url",
    "skipped",
    "skip_datetime",
    "is_relevant",
    "tags_selected",
    "tags_added",
    "tags_final",
    "labels_selected",
    "labels_added",
    "labels_final",
    "coder_notes",
]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_string() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def project_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def safe_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    return str(value)


def load_json(path: Path, default: Any) -> Any:
    if not path.is_file():
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_app_config() -> dict[str, Any]:
    config = DEFAULT_CONFIG.copy()
    config.update(load_json(CONFIG_PATH, {}))
    return config


def load_options(path: Path) -> list[str]:
    options = load_json(path, [])
    if not isinstance(options, list):
        return []
    return [str(option) for option in options]


def collection_date_from_config(config: dict[str, Any]) -> str:
    configured_date = str(config.get("collection_date") or "").strip()
    return configured_date or os.getenv("ANALYSIS_DATE") or today_string()


def processed_csv_path(config: dict[str, Any], collection_date: str) -> Path:
    data_dir = project_path(config["processed_data_dir"])
    filename = str(config.get("processed_data_filename") or "enriched_results_sample.csv")
    return data_dir / collection_date / filename


def annotations_root(config: dict[str, Any]) -> Path:
    return project_path(config["annotations_dir"])


def safe_coder_slug(coder_name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", coder_name.strip()).strip("_")
    return slug or "coder"


def annotation_file_path(root: Path, coder_name: str) -> Path:
    return root / "raw" / f"{safe_coder_slug(coder_name)}_annotations.csv"


def stable_annotation_id(coder_name: str, item_id: str) -> str:
    value = f"{coder_name}|{item_id}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()[:16]


def split_values(value: Any) -> list[str]:
    parts = re.split(r"[;,\n]", safe_value(value))
    return [part.strip() for part in parts if part.strip()]


def unique_values(values: list[str]) -> list[str]:
    seen = set()
    unique = []
    for value in values:
        key = value.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(value)
    return unique


def join_values(values: list[str]) -> str:
    return "; ".join(unique_values(values))


@st.cache_data
def load_posts(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(ttl=300)
def fetch_thread(post_uri: str) -> tuple[dict[str, Any] | None, str]:
    if not post_uri:
        return None, "Missing post_uri."

    try:
        response = requests.get(
            PUBLIC_BSKY_THREAD_API,
            params={
                "uri": post_uri,
                "depth": THREAD_DEPTH,
                "parentHeight": THREAD_PARENT_HEIGHT,
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        return None, str(e)

    return response.json().get("thread"), ""


def ensure_post_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    missing_columns = [column for column in POST_COLUMNS if column not in df.columns]
    df = df.where(pd.notna(df), "").astype(str).copy()
    for column in missing_columns:
        df[column] = ""
    ordered_columns = POST_COLUMNS + [
        column for column in df.columns if column not in POST_COLUMNS
    ]
    return df[ordered_columns], missing_columns


def load_annotations(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)

    df = pd.read_csv(path, dtype=str).fillna("")
    for column in ANNOTATION_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    return df[ANNOTATION_COLUMNS]


def latest_annotation(
    annotations: pd.DataFrame,
    coder_name: str,
    item_id: str,
) -> dict[str, Any]:
    if annotations.empty:
        return {}

    matches = annotations[
        (annotations["coder_name"].astype(str) == coder_name)
        & (annotations["item_id"].astype(str) == item_id)
    ].copy()
    if matches.empty:
        return {}

    matches = matches.sort_values("annotation_datetime")
    return matches.iloc[-1].to_dict()


def latest_rows_for_coder(annotations: pd.DataFrame, coder_name: str) -> pd.DataFrame:
    if annotations.empty:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)

    coder_rows = annotations[
        annotations["coder_name"].astype(str) == coder_name
    ].copy()
    if coder_rows.empty:
        return pd.DataFrame(columns=ANNOTATION_COLUMNS)

    coder_rows = coder_rows.sort_values("annotation_datetime")
    coder_rows = coder_rows.drop_duplicates(["coder_name", "item_id"], keep="last")
    return coder_rows[ANNOTATION_COLUMNS].reset_index(drop=True)


def handled_item_ids(annotations: pd.DataFrame, coder_name: str) -> set[str]:
    if annotations.empty:
        return set()

    coder_rows = latest_rows_for_coder(annotations, coder_name)
    return set(coder_rows["item_id"].dropna().astype(str))


def progress_counts(
    posts: pd.DataFrame,
    annotations: pd.DataFrame,
    coder_name: str,
) -> tuple[int, int, int]:
    coder_rows = latest_rows_for_coder(annotations, coder_name)
    handled_count = coder_rows["item_id"].dropna().astype(str).nunique()
    skipped_count = (
        coder_rows["skipped"].astype(str).str.lower().eq("true").sum()
        if not coder_rows.empty
        else 0
    )
    annotated_count = max(handled_count - int(skipped_count), 0)
    remaining_count = max(len(posts) - handled_count, 0)
    return annotated_count, int(skipped_count), remaining_count


def next_unhandled_post(posts: pd.DataFrame, handled: set[str]) -> pd.Series | None:
    for _, row in posts.iterrows():
        item_id = safe_value(row.get("item_id"))
        if item_id and item_id not in handled:
            return row
    return None


def post_for_item_id(posts: pd.DataFrame, item_id: str) -> pd.Series | None:
    """Return the post with the requested item ID."""
    for _, row in posts.iterrows():
        if safe_value(row.get("item_id")) == item_id:
            return row
    return None


def post_number_for_item_id(posts: pd.DataFrame, item_id: str) -> int:
    """Return the one-based position of an item in the annotation input."""
    for position, (_, row) in enumerate(posts.iterrows(), start=1):
        if safe_value(row.get("item_id")) == item_id:
            return position
    return 0


def previous_handled_item_id(
    posts: pd.DataFrame,
    handled: set[str],
    before_item_id: str | None = None,
) -> str | None:
    """Return the nearest handled item before another post in input order."""
    ordered_item_ids = [
        safe_value(row.get("item_id"))
        for _, row in posts.iterrows()
        if safe_value(row.get("item_id"))
    ]
    before_index = len(ordered_item_ids)

    if before_item_id:
        try:
            before_index = ordered_item_ids.index(before_item_id)
        except ValueError:
            return None

    for item_id in reversed(ordered_item_ids[:before_index]):
        if item_id in handled:
            return item_id
    return None


def next_item_id(posts: pd.DataFrame, after_item_id: str) -> str | None:
    """Return the item immediately after another post in input order."""
    ordered_item_ids = [
        safe_value(row.get("item_id"))
        for _, row in posts.iterrows()
        if safe_value(row.get("item_id"))
    ]
    try:
        next_index = ordered_item_ids.index(after_item_id) + 1
    except ValueError:
        return None

    return ordered_item_ids[next_index] if next_index < len(ordered_item_ids) else None


def options_from_annotations(annotations: pd.DataFrame, column: str) -> list[str]:
    if annotations.empty or column not in annotations.columns:
        return []

    values = []
    for value in annotations[column].dropna():
        values.extend(split_values(value))

    return unique_values(values)


def save_annotation(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    annotations = load_annotations(path)
    updated = pd.concat([annotations, pd.DataFrame([row])], ignore_index=True)
    updated = updated.where(pd.notna(updated), "").astype(str)
    updated = updated.sort_values("annotation_datetime")
    updated = updated.drop_duplicates(["coder_name", "item_id"], keep="last")
    updated = updated[ANNOTATION_COLUMNS]
    updated.to_csv(path, index=False)


def build_annotation_row(
    post: pd.Series,
    coder_name: str,
    clean_coding_mode: bool,
    skipped: bool,
    is_relevant: str,
    tags_selected: list[str],
    tags_added: list[str],
    labels_selected: list[str],
    labels_added: list[str],
    coder_notes: str,
) -> dict[str, Any]:
    item_id = safe_value(post.get("item_id"))
    now = utc_now_iso()
    if skipped:
        tags_selected = []
        tags_added = []
        labels_selected = []
        labels_added = []

    row = {
        "annotation_id": stable_annotation_id(coder_name, item_id),
        "annotation_datetime": now,
        "coder_name": coder_name,
        "clean_coding_mode": clean_coding_mode,
        "item_id": item_id,
        "post_url": safe_value(post.get("post_url")),
        "skipped": skipped,
        "skip_datetime": now if skipped else "",
        "is_relevant": "" if skipped else is_relevant,
        "tags_selected": join_values(tags_selected),
        "tags_added": join_values(tags_added),
        "tags_final": join_values(tags_selected + tags_added),
        "labels_selected": join_values(labels_selected),
        "labels_added": join_values(labels_added),
        "labels_final": join_values(labels_selected + labels_added),
        "coder_notes": coder_notes,
    }

    return {column: row.get(column, "") for column in ANNOTATION_COLUMNS}


def rerun_app() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
        return
    st.experimental_rerun()


def open_thread_button(post: pd.Series) -> None:
    post_url = safe_value(post.get("post_url"))
    if not post_url:
        st.warning("No post_url available for this row.")
        return

    st.link_button("Open full Bluesky thread", post_url, use_container_width=True)


def post_url_from_view(post: dict[str, Any]) -> str:
    uri = safe_value(post.get("uri"))
    author_handle = safe_value((post.get("author") or {}).get("handle"))
    rkey = uri.split("/")[-1]

    if not author_handle or not rkey:
        return ""

    return f"https://bsky.app/profile/{author_handle}/post/{rkey}"


def post_text_from_view(post: dict[str, Any]) -> str:
    record = post.get("record") or {}
    return safe_value(record.get("text"))


def post_datetime_from_view(post: dict[str, Any]) -> str:
    record = post.get("record") or {}
    return safe_value(record.get("createdAt"))


def render_thread_node(
    node: dict[str, Any],
    target_uri: str,
    level: int = 0,
    include_replies: bool = True,
) -> None:
    node_type = safe_value(node.get("$type"))
    post = node.get("post")

    if not post:
        message = "Thread item unavailable"
        if "blocked" in node_type.lower():
            message = "Blocked post"
        elif "notFound" in node_type:
            message = "Post not found"
        st.caption(("↳ " * level) + message)
        return

    post_uri = safe_value(post.get("uri"))
    author = post.get("author") or {}
    author_handle = safe_value(author.get("handle"))
    display_name = safe_value(author.get("displayName"))
    created_at = post_datetime_from_view(post)
    text = post_text_from_view(post)
    post_url = post_url_from_view(post)
    is_target = post_uri == target_uri
    prefix = "↳ " * level

    with st.container(border=True):
        author_label = f"{display_name} (@{author_handle})" if display_name else f"@{author_handle}"
        if is_target:
            st.markdown(f"{prefix}**Current post · {author_label}**")
        else:
            st.markdown(f"{prefix}**{author_label}**")
        st.caption(created_at)
        st.write(text or "_No text._")
        st.caption(
            f"likes {post.get('likeCount', 0)} · "
            f"reposts {post.get('repostCount', 0)} · "
            f"replies {post.get('replyCount', 0)} · "
            f"quotes {post.get('quoteCount', 0)}"
        )
        if post_url:
            st.markdown(f"[Open this post]({post_url})")

    if include_replies:
        for reply in node.get("replies") or []:
            render_thread_node(reply, target_uri, level + 1)


def parent_chain(node: dict[str, Any]) -> list[dict[str, Any]]:
    parent = node.get("parent")
    if not isinstance(parent, dict):
        return []

    return parent_chain(parent) + [parent]


def render_thread(post: pd.Series) -> None:
    post_uri = safe_value(post.get("post_uri"))
    thread, error = fetch_thread(post_uri)

    if error:
        st.warning(f"Could not load Bluesky thread: {error}")
        return

    if not thread:
        st.warning("Bluesky returned an empty thread.")
        return

    st.markdown("#### Thread")
    render_thread_node(thread, post_uri)

    parents = parent_chain(thread)
    if parents:
        st.markdown("#### Parent context")
        for parent in parents:
            render_thread_node(parent, post_uri, include_replies=False)


def display_post(post: pd.Series) -> None:
    open_thread_button(post)
    render_thread(post)


def render_setup_screen(collection_date: str, input_label: str, post_count: int) -> None:
    st.caption(f"Collection date: {collection_date} · Posts loaded: {post_count}")
    st.caption(f"Input: {input_label}")
    st.info(
        "You will review Bluesky posts one at a time. For each post, decide whether "
        "it is relevant, add useful tags, and note any label names mentioned. Use "
        "Skip only when you cannot make a judgment; otherwise submit the post even "
        "if it is irrelevant or unclear."
    )

    with st.form("coder_setup_form"):
        coder_name = st.text_input("Coder name")
        clean_coding_mode = st.radio(
            "Clean coding mode?",
            ["Yes", "No"],
            horizontal=True,
            help="Clean mode starts with no shared tag suggestions. Your own tags will appear after you create them.",
        ) == "Yes"
        started = st.form_submit_button("Start annotation")

    if not started:
        st.stop()

    coder_name = coder_name.strip()
    if not coder_name:
        st.error("Please enter a coder name.")
        st.stop()

    st.session_state["coder_name"] = coder_name
    st.session_state["clean_coding_mode"] = clean_coding_mode
    st.session_state["setup_complete"] = True
    rerun_app()


def disable_enter_submit() -> None:
    components.html(
        """
        <script>
        window.parent.document.addEventListener("keydown", function(event) {
          const tagName = event.target.tagName.toLowerCase();
          if (event.key === "Enter" && tagName === "input") {
            event.preventDefault();
          }
        }, true);
        </script>
        """,
        height=0,
    )


def reset_setup() -> None:
    for key in [
        "coder_name",
        "clean_coding_mode",
        "setup_complete",
        "review_item_id",
    ]:
        st.session_state.pop(key, None)
    rerun_app()


def render_option_selector(
    label: str,
    options: list[str],
    helper_empty: str,
    default_selected: list[str],
    key_prefix: str,
) -> tuple[list[str], str]:
    if options:
        selected = st.multiselect(
            label,
            options,
            default=[value for value in default_selected if value in options],
            key=f"{key_prefix}_selected",
        )
    else:
        selected = []
        st.caption(helper_empty)

    added = st.text_area(
        f"Add {label.lower()}",
        help="Add multiple values separated by new lines, commas, or semicolons.",
        placeholder="One per line, or comma-separated",
        height=110,
        key=f"{key_prefix}_added",
    )

    return selected, added


def main() -> None:
    st.set_page_config(page_title="sTechLab Labeler Annotation", layout="wide")
    disable_enter_submit()
    st.title("sTechLab Labeler Content Annotation")

    config = load_app_config()
    collection_date = collection_date_from_config(config)
    input_path = processed_csv_path(config, collection_date)
    annotation_root = annotations_root(config)
    configured_tags = load_options(TAGS_PATH)

    if not input_path.is_file():
        st.error(f"Processed annotation CSV not found: {input_path}")
        st.info("Add the sample CSV at the configured path, then restart the app.")
        st.stop()

    posts = load_posts(str(input_path))
    input_label = f"{collection_date}/{input_path.name}"

    posts, missing_columns = ensure_post_columns(posts)
    if missing_columns:
        st.warning("Missing columns added as blanks: " + ", ".join(missing_columns))

    if not st.session_state.get("setup_complete"):
        render_setup_screen(collection_date, input_label, len(posts))

    coder_name = st.session_state["coder_name"]
    clean_coding_mode = bool(st.session_state["clean_coding_mode"])

    annotation_path = annotation_file_path(annotation_root, coder_name)
    all_annotations = load_annotations(annotation_path)
    coder_annotations = latest_rows_for_coder(all_annotations, coder_name)
    handled = handled_item_ids(coder_annotations, coder_name)
    annotated_count, skipped_count, remaining_count = progress_counts(
        posts,
        all_annotations,
        coder_name,
    )

    coder_tags = options_from_annotations(coder_annotations, "tags_final")
    coder_labels = options_from_annotations(coder_annotations, "labels_final")
    tags_options = coder_tags if clean_coding_mode else unique_values(configured_tags + coder_tags)
    labels_options = coder_labels

    st.sidebar.markdown("### Session")
    st.sidebar.write(f"Coder: {coder_name}")
    st.sidebar.write(f"Mode: {'Clean' if clean_coding_mode else 'Codebook'}")
    if st.sidebar.button("Change coder / mode"):
        reset_setup()
    st.sidebar.markdown("### Progress")
    st.sidebar.metric("Annotated", f"{annotated_count} / {len(posts)}")
    st.sidebar.metric("Skipped", skipped_count)
    st.sidebar.metric("Remaining", remaining_count)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Finish")
    if not coder_annotations.empty:
        st.sidebar.success("Download your CSV before closing the app.")
        st.sidebar.download_button(
            "⬇ Download my annotations CSV",
            coder_annotations.to_csv(index=False).encode("utf-8"),
            file_name=f"{safe_coder_slug(coder_name)}_annotations.csv",
            mime="text/csv",
            use_container_width=True,
            type="primary",
        )
    else:
        st.sidebar.info("Your download button will appear after your first save.")

    current_post = next_unhandled_post(posts, handled)
    review_item_id = safe_value(st.session_state.get("review_item_id"))
    review_post = (
        post_for_item_id(posts, review_item_id)
        if review_item_id and review_item_id in handled
        else None
    )
    if review_item_id and review_post is None:
        st.session_state.pop("review_item_id", None)

    is_reviewing = review_post is not None
    post = review_post if is_reviewing else current_post

    if post is None:
        st.success(
            "No posts left for this coder. Download your annotations CSV from "
            "the sidebar and send it back before closing the app."
        )
        previous_item_id = previous_handled_item_id(posts, handled)
        if previous_item_id and st.button("Back to previous answer"):
            st.session_state["review_item_id"] = previous_item_id
            rerun_app()
        st.stop()

    item_id = safe_value(post.get("item_id"))
    existing = latest_annotation(coder_annotations, coder_name, item_id)
    current_number = post_number_for_item_id(posts, item_id)
    previous_item_id = previous_handled_item_id(posts, handled, item_id)

    if is_reviewing:
        st.info(
            "You are editing a saved answer. Changes are saved only when you "
            "select Save changes or Mark skipped."
        )
        return_label = (
            "Return to current post" if current_post is not None else "Return to completion"
        )
        if st.button(return_label):
            st.session_state.pop("review_item_id", None)
            rerun_app()

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader(f"Post {current_number} of {len(posts)}")
        display_post(post)

    with right:
        st.subheader("Edit annotation" if is_reviewing else "Coding form")
        if coder_tags:
            st.caption("Your tags so far: " + ", ".join(coder_tags))
        if coder_labels:
            st.caption("Your labels so far: " + ", ".join(coder_labels))

        existing_tags = split_values(existing.get("tags_final"))
        existing_labels = split_values(existing.get("labels_final"))
        post_tags_options = unique_values(tags_options + existing_tags)
        post_labels_options = unique_values(labels_options + existing_labels)
        existing_version = safe_value(existing.get("annotation_datetime"))
        version_hash = hashlib.sha256(existing_version.encode("utf-8")).hexdigest()[:8]
        key_prefix = (
            f"annotation_{safe_coder_slug(coder_name)}_{item_id}_{version_hash}"
        )

        with st.form(f"{key_prefix}_form"):
            relevant_options = ["Yes", "No", "Unclear"]
            existing_relevance = safe_value(existing.get("is_relevant"))
            relevance_index = (
                relevant_options.index(existing_relevance)
                if existing_relevance in relevant_options
                else None if is_reviewing else 0
            )
            is_relevant = st.radio(
                "Is the post relevant?",
                relevant_options,
                index=relevance_index,
                horizontal=True,
                key=f"{key_prefix}_relevance",
            )

            tags_selected, tags_added_text = render_option_selector(
                "Tags",
                post_tags_options,
                "No saved tags yet. Add the first one below.",
                existing_tags,
                f"{key_prefix}_tags",
            )
            labels_selected, labels_added_text = render_option_selector(
                "Labels mentioned",
                post_labels_options,
                "No saved labels yet. Add the first one below.",
                existing_labels,
                f"{key_prefix}_labels",
            )

            coder_notes = st.text_area(
                "Notes / skip reason (optional)",
                value=safe_value(existing.get("coder_notes")),
                key=f"{key_prefix}_notes",
            )

            submit_col, skip_col, back_col = st.columns(3)
            submit_label = "Save changes" if is_reviewing else "Submit"
            skip_label = "Mark skipped" if is_reviewing else "Skip"
            submitted = submit_col.form_submit_button(submit_label)
            skipped = skip_col.form_submit_button(skip_label)
            went_back = back_col.form_submit_button(
                "Back",
                disabled=previous_item_id is None,
            )

        if went_back and previous_item_id:
            st.session_state["review_item_id"] = previous_item_id
            rerun_app()

        if submitted and not is_relevant:
            st.error("Choose Yes, No, or Unclear before saving this annotation.")
        elif submitted or skipped:
            row = build_annotation_row(
                post=post,
                coder_name=coder_name,
                clean_coding_mode=clean_coding_mode,
                skipped=skipped,
                is_relevant=is_relevant,
                tags_selected=tags_selected,
                tags_added=split_values(tags_added_text),
                labels_selected=labels_selected,
                labels_added=split_values(labels_added_text),
                coder_notes=coder_notes,
            )
            save_annotation(annotation_path, row)
            next_id = next_item_id(posts, item_id) if is_reviewing else None
            if next_id and next_id in handled:
                st.session_state["review_item_id"] = next_id
            else:
                st.session_state.pop("review_item_id", None)
            st.success("Saved. Remember to download your CSV when finished.")
            rerun_app()


if __name__ == "__main__":
    main()
