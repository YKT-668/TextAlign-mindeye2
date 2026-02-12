#!/usr/bin/env python3
"""Index all paper asset files/symlinks under /mnt/work/paper_assets/ALL.

Read-only scan. Generates:
- results/paper_inventory/ALL_SYMLINKS_INDEX.md
- results/paper_inventory/ALL_SYMLINKS_INDEX.tsv
- results/paper_inventory/ALL_REALPATHS_UNION.txt

Also prints a brief stdout summary.

Notes:
- We index *files and symlinks* (not directories).
- For symlinks, `symlink_target` is the raw os.readlink() string.
- `target_exists` is computed by resolving relative targets against the link's parent dir.
- `size_bytes` and `mtime` are taken from os.stat(path) (follows symlinks). If broken, they're NA.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


ALL_ROOT_DEFAULT = Path("/mnt/work/paper_assets/ALL")


@dataclass
class Entry:
    module: str
    rel_path: str  # relative to ALL
    abs_path: str
    is_symlink: bool
    symlink_target: str
    target_exists: bool
    file_type: str
    size_bytes: Optional[int]
    mtime_iso: str
    realpath: Optional[str]


def iso_from_mtime(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def file_type_for(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    if ext in {"png", "csv", "md", "json", "txt", "log"}:
        return ext
    return "other"


def resolve_link_target(link_path: Path, raw_target: str) -> Path:
    t = Path(raw_target)
    if t.is_absolute():
        return t
    return (link_path.parent / t).resolve()


def module_for(rel_posix: str) -> str:
    # rel_posix is relative to ALL
    parts = [p for p in rel_posix.split("/") if p]
    if not parts:
        return "(root)"
    return parts[0]


def iter_entries(all_root: Path) -> List[Entry]:
    entries: List[Entry] = []

    for dirpath, dirnames, filenames in os.walk(all_root, followlinks=False):
        # We only index files/symlinks; directories are implicit.
        for name in filenames:
            p = Path(dirpath) / name
            try:
                rel = p.relative_to(all_root).as_posix()
            except Exception:
                # Should not happen, but keep safe.
                continue

            is_link = os.path.islink(p)
            raw_target = os.readlink(p) if is_link else ""

            target_exists = True
            resolved_target: Optional[Path] = None
            if is_link:
                resolved_target = resolve_link_target(p, raw_target)
                target_exists = os.path.exists(resolved_target)

            # Stats: requirement says use os.stat (follows symlink)
            size: Optional[int]
            mtime_iso: str
            try:
                st = os.stat(p)
                size = int(st.st_size)
                mtime_iso = iso_from_mtime(st.st_mtime)
            except Exception:
                size = None
                mtime_iso = "NA"

            ftype = file_type_for(p)

            # realpath union: include only if it resolves to an existing file
            realpath: Optional[str] = None
            if is_link:
                if target_exists and resolved_target is not None:
                    realpath = str(resolved_target)
            else:
                # regular file
                realpath = str(p.resolve())

            entries.append(
                Entry(
                    module=module_for(rel),
                    rel_path=rel,
                    abs_path=str(p),
                    is_symlink=is_link,
                    symlink_target=raw_target,
                    target_exists=bool(target_exists),
                    file_type=ftype,
                    size_bytes=size,
                    mtime_iso=mtime_iso,
                    realpath=realpath,
                )
            )

    # Deterministic order
    entries.sort(key=lambda e: (e.module, e.rel_path))
    return entries


def write_tsv(out_path: Path, entries: List[Entry]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(
            "\t".join(
                [
                    "module",
                    "link_path_rel_all",
                    "is_symlink",
                    "symlink_target",
                    "target_exists",
                    "file_type",
                    "size_bytes",
                    "mtime_iso",
                ]
            )
            + "\n"
        )
        for e in entries:
            f.write(
                "\t".join(
                    [
                        e.module,
                        e.rel_path,
                        "True" if e.is_symlink else "False",
                        e.symlink_target,
                        "True" if e.target_exists else "False",
                        e.file_type,
                        "NA" if e.size_bytes is None else str(e.size_bytes),
                        e.mtime_iso,
                    ]
                )
                + "\n"
            )


def module_stats(entries: List[Entry]) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for e in entries:
        s = stats.setdefault(
            e.module,
            {
                "count_total": 0,
                "count_png": 0,
                "count_csv": 0,
                "count_md": 0,
                "broken_symlink_count": 0,
            },
        )
        s["count_total"] += 1
        if e.file_type == "png":
            s["count_png"] += 1
        if e.file_type == "csv":
            s["count_csv"] += 1
        if e.file_type == "md":
            s["count_md"] += 1
        if e.is_symlink and not e.target_exists:
            s["broken_symlink_count"] += 1
    return stats


def write_md(out_path: Path, entries: List[Entry]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stats = module_stats(entries)

    lines: List[str] = []
    lines.append("# ALL symlinks/files index")
    lines.append("")
    lines.append(f"- scanned_root: {ALL_ROOT_DEFAULT}")
    lines.append(f"- generated_utc: {datetime.now(timezone.utc).isoformat().replace('+00:00','Z')}")
    lines.append("")

    by_module: Dict[str, List[Entry]] = {}
    for e in entries:
        by_module.setdefault(e.module, []).append(e)

    for module in sorted(by_module.keys()):
        s = stats.get(module, {})
        lines.append(f"## {module}")
        lines.append("")
        lines.append(
            "- stats: "
            + ", ".join(
                [
                    f"count_total={s.get('count_total', 0)}",
                    f"count_png={s.get('count_png', 0)}",
                    f"count_csv={s.get('count_csv', 0)}",
                    f"count_md={s.get('count_md', 0)}",
                    f"broken_symlink_count={s.get('broken_symlink_count', 0)}",
                ]
            )
        )
        lines.append("")
        lines.append(
            "| link_path (rel ALL) | is_symlink | symlink_target | target_exists | file_type | size_bytes | mtime (ISO) |"
        )
        lines.append("|---|---|---|---|---|---:|---|")

        for e in by_module[module]:
            lines.append(
                "| {p} | {islink} | {tgt} | {exists} | {ft} | {sz} | {mt} |".format(
                    p=e.rel_path,
                    islink="True" if e.is_symlink else "False",
                    tgt=(e.symlink_target.replace("|", "\\|") if e.symlink_target else ""),
                    exists="True" if e.target_exists else "False",
                    ft=e.file_type,
                    sz="NA" if e.size_bytes is None else str(e.size_bytes),
                    mt=e.mtime_iso,
                )
            )
        lines.append("")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_realpaths_union(out_path: Path, entries: List[Entry]) -> List[str]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    realpaths = sorted({e.realpath for e in entries if e.realpath})
    out_path.write_text("\n".join(realpaths) + "\n", encoding="utf-8")
    return realpaths


def main(argv: Sequence[str]) -> int:
    ap = argparse.ArgumentParser(description="Index /mnt/work/paper_assets/ALL into MD/TSV/realpaths union")
    ap.add_argument("--all-root", type=str, default=str(ALL_ROOT_DEFAULT))
    ap.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    args = ap.parse_args(argv)

    all_root = Path(args.all_root).resolve()
    repo_root = Path(args.repo_root).resolve()

    out_dir = repo_root / "results/paper_inventory"
    out_md = out_dir / "ALL_SYMLINKS_INDEX.md"
    out_tsv = out_dir / "ALL_SYMLINKS_INDEX.tsv"
    out_union = out_dir / "ALL_REALPATHS_UNION.txt"

    entries = iter_entries(all_root)
    write_md(out_md, entries)
    write_tsv(out_tsv, entries)
    realpaths = write_realpaths_union(out_union, entries)

    total_links = sum(1 for e in entries if e.is_symlink)
    broken_links = sum(1 for e in entries if e.is_symlink and not e.target_exists)
    total_targets_unique = len(realpaths)

    # Top 5 modules by broken links
    broken_by_module: Dict[str, int] = {}
    for e in entries:
        if e.is_symlink and not e.target_exists:
            broken_by_module[e.module] = broken_by_module.get(e.module, 0) + 1

    top5 = sorted(broken_by_module.items(), key=lambda kv: (-kv[1], kv[0]))[:5]

    print("SUMMARY")
    print(f"- total_links: {total_links}")
    print(f"- broken_links: {broken_links}")
    print(f"- total_targets_unique: {total_targets_unique}")
    print("- broken_links_top5_by_module:")
    if not top5:
        print("  - none")
    else:
        for m, c in top5:
            print(f"  - {m}: {c}")

    print("WROTE")
    print(f"- {out_md}")
    print(f"- {out_tsv}")
    print(f"- {out_union}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(os.sys.argv[1:]))
