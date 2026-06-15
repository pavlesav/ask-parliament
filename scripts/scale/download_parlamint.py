"""Download ParlaMint 5.0 country corpora from CLARIN.SI (resumable).

ParlaMint 5.0 (handle 11356/2004, CC BY 4.0) is published as one .tgz per
country/region. This script discovers the real download URLs from the repository
page, fetches the requested countries, and extracts only the plain-text +
TSV-metadata subtree (`ParlaMint-{CC}.txt/`) — the TEI XML is not needed and is
skipped to save disk.

Resumable: a country whose `{CC}/.extracted` marker exists is skipped. Rerun
freely; pass --force to redo one. Archives are deleted after extraction unless
--keep-archives is given.

Examples:
  python scripts/scale/download_parlamint.py --list          # show URLs, exit
  python scripts/scale/download_parlamint.py --countries LV   # one country
  python scripts/scale/download_parlamint.py                  # all 29
"""
from __future__ import annotations

import argparse
import logging
import re
import tarfile
import time
from pathlib import Path

import requests

from ask_parliament.config import (
    PARLAMINT_COUNTRIES,
    PARLAMINT_HANDLE,
    PARLAMINT_PAGE,
    RAW_DIR,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("download_parlamint")

# Matches e.g. bitstream/handle/11356/2004/ParlaMint-ES-CT.tgz?sequence=9
_LINK = re.compile(r"bitstream/handle/[^/]+/[^/]+/(ParlaMint-([A-Z]{2}(?:-[A-Z]{2})?)\.tgz)\?sequence=(\d+)")


def discover_urls() -> dict[str, str]:
    """Scrape the repository page for the per-country .tgz download URLs."""
    log.info("Discovering download URLs from %s", PARLAMINT_PAGE)
    html = requests.get(PARLAMINT_PAGE, timeout=60).text
    prefix = PARLAMINT_PAGE.split("/handle/")[0] + "/"  # .../repository/xmlui/
    urls: dict[str, str] = {}
    for fname, code, seq in _LINK.findall(html):
        urls[code] = f"{prefix}bitstream/handle/{PARLAMINT_HANDLE}/{fname}?sequence={seq}&isAllowed=y"
    return urls


def download(url: str, dest: Path) -> None:
    """Stream a URL to `dest` via a .part temp file, verifying Content-Length."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done, next_log = 0, 50 * 1024 * 1024
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
                done += len(chunk)
                if done >= next_log:
                    log.info("  ... %d MB%s", done // (1024 * 1024),
                             f" / {total // (1024 * 1024)} MB" if total else "")
                    next_log += 50 * 1024 * 1024
    if total and tmp.stat().st_size != total:
        raise IOError(f"{dest.name}: size mismatch (got {tmp.stat().st_size}, expected {total})")
    tmp.replace(dest)


def extract_txt_subtree(archive: Path, country: str, out_dir: Path) -> int:
    """Extract only the `ParlaMint-{CC}.txt/` members; return the file count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = f"ParlaMint-{country}.txt/"
    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if m.name.startswith(wanted)]
        if not members:
            raise RuntimeError(f"{archive.name}: no '{wanted}' members found")
        tar.extractall(path=out_dir, members=members, filter="data")
    return sum(1 for m in members if m.isfile())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--countries", help="comma-separated codes (default: all 29)")
    p.add_argument("--list", action="store_true", help="print discovered URLs and exit")
    p.add_argument("--force", action="store_true", help="re-download/extract even if already done")
    p.add_argument("--keep-archives", action="store_true", help="keep .tgz files after extraction")
    a = p.parse_args()

    urls = discover_urls()
    if a.list:
        for cc in PARLAMINT_COUNTRIES:
            print(f"{cc:6} {urls.get(cc, '(not found)')}")
        log.info("Discovered %d/%d country archives",
                 sum(c in urls for c in PARLAMINT_COUNTRIES), len(PARLAMINT_COUNTRIES))
        return

    countries = ([c.strip() for c in a.countries.split(",") if c.strip()]
                 if a.countries else list(PARLAMINT_COUNTRIES))
    archives_dir = RAW_DIR / "_archives"

    for cc in countries:
        if cc not in urls:
            log.warning("%s: no download URL found — skipping", cc)
            continue
        out_dir = RAW_DIR / cc
        marker = out_dir / ".extracted"
        if marker.exists() and not a.force:
            log.info("%s: already downloaded — skipping (use --force to redo)", cc)
            continue
        archive = archives_dir / f"ParlaMint-{cc}.tgz"
        t0 = time.time()
        if not archive.exists() or a.force:
            log.info("%s: downloading ...", cc)
            download(urls[cc], archive)
        log.info("%s: extracting text subtree ...", cc)
        n = extract_txt_subtree(archive, cc, out_dir)
        marker.write_text(f"ParlaMint {PARLAMINT_HANDLE} {cc}: {n} files, "
                          f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if not a.keep_archives:
            archive.unlink(missing_ok=True)
        log.info("%s: done (%d files, %.1f s)", cc, n, time.time() - t0)


if __name__ == "__main__":
    main()
