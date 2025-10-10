"""Page handling helpers for PDF processing."""

from __future__ import annotations

from typing import Iterable, List, Optional, Set


def resolve_page_indices(pages: Optional[Iterable[int]], total_pages: int) -> List[int]:
    """Convert requested pages to zero-based indices within document bounds.

    Args:
        pages: Iterable of 1-based page numbers supplied by callers. ``None``
            or an empty iterable selects every available page. Values less than
            ``1`` default to the first page so that legacy zero-based inputs do
            not underflow.
        total_pages: Total number of pages available in the document.

    Returns:
        A list of zero-based page indices, preserving the original ordering and
        removing duplicates while clamping values to the available range.
    """

    if total_pages <= 0:
        return []

    if not pages:
        return list(range(total_pages))

    resolved: List[int] = []
    seen: Set[int] = set()
    max_index = total_pages - 1

    for page in pages:
        if page is None:
            continue

        index = page - 1 if page > 0 else 0
        if index > max_index:
            index = max_index

        if index not in seen:
            resolved.append(index)
            seen.add(index)

    return resolved
