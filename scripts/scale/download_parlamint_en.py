"""Download the ParlaMint-en.ana 5.0 English machine translations (resumable).

The English edition (handle 11356/2006, CC BY 4.0) ships one `ParlaMint-{CC}-en.ana.tgz`
per country. Each archive is large (it also carries TEI/CoNLL-U/vertical annotation),
but it bundles a derived `ParlaMint-{CC}-en.txt/` plain-text subtree — the same
`ID<tab>text` layout as the native corpus, keyed by the *native* utterance ids — which
is all we need to show cited speeches in English. This script downloads each archive,
extracts ONLY that text subtree (skipping the heavy annotation), and deletes the archive,
so peak disk stays small even though the downloads are big.

Resumable: a country whose `{CC}/.extracted` marker exists is skipped. Rerun freely;
pass --force to redo one.

Examples:
  python scripts/scale/download_parlamint_en.py --list          # show URLs, exit
  python scripts/scale/download_parlamint_en.py --countries LV   # one country
  python scripts/scale/download_parlamint_en.py                  # all 29
"""
from __future__ import annotations

import argparse
import logging
import re
import tarfile
import time
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ask_parliament.config import (
    PARLAMINT_COUNTRIES,
    PARLAMINT_EN_HANDLE,
    PARLAMINT_EN_PAGE,
    RAW_EN_DIR,
)

logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
log = logging.getLogger("download_parlamint_en")

# CLARIN.SI intermittently drops programmatic connects (while browsers get through), so use
# a browser User-Agent and short, frequently-retried connects (urllib3 retries the TCP connect
# with backoff) rather than one long 120 s wait — we cycle through the server's bad moments.
_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
_CONNECT_TIMEOUT = 30  # seconds per connect attempt (urllib3 retries cover transient drops)


def _make_session() -> requests.Session:
    s = requests.Session()
    # Look like a browser (CLARIN lets browsers through while sometimes refusing bare clients).
    s.headers.update({
        "User-Agent": _UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    retry = Retry(total=6, connect=6, read=3, backoff_factor=3,
                  status_forcelist=[429, 500, 502, 503, 504], respect_retry_after_header=True)
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


_SESSION = _make_session()

# Matches e.g. bitstream/handle/11356/2006/ParlaMint-ES-CT-en.ana.tgz?sequence=9
_LINK = re.compile(
    r"bitstream/handle/[^/]+/[^/]+/(ParlaMint-([A-Z]{2}(?:-[A-Z]{2})?)-en\.ana\.tgz)\?sequence=(\d+)"
)


def _retry(fn, what: str, tries: int = 4, backoff: int = 15):
    """Run fn(), retrying transient network/IO errors with linear backoff. CLARIN.SI is
    occasionally unreachable for a stretch, so a long unattended pull shouldn't die on one."""
    for i in range(1, tries + 1):
        try:
            return fn()
        except (requests.RequestException, IOError) as err:
            if i == tries:
                raise
            log.warning("%s failed (%s); retry %d/%d in %ds", what, err, i, tries, backoff * i)
            time.sleep(backoff * i)


def discover_urls() -> dict[str, str]:
    """Scrape the English record page for the per-country .ana.tgz download URLs."""
    log.info("Discovering download URLs from %s", PARLAMINT_EN_PAGE)
    html = _retry(lambda: _SESSION.get(PARLAMINT_EN_PAGE, timeout=(_CONNECT_TIMEOUT, 120)),
                  "URL discovery").text
    prefix = PARLAMINT_EN_PAGE.split("/handle/")[0] + "/"  # .../repository/xmlui/
    urls: dict[str, str] = {}
    for fname, code, seq in _LINK.findall(html):
        urls[code] = f"{prefix}bitstream/handle/{PARLAMINT_EN_HANDLE}/{fname}?sequence={seq}&isAllowed=y"
    return urls


def download(url: str, dest: Path) -> None:
    """Stream a URL to `dest` via a .part temp file, verifying Content-Length."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    with _SESSION.get(url, stream=True, timeout=(_CONNECT_TIMEOUT, 300)) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        done, next_log = 0, 100 * 1024 * 1024
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                fh.write(chunk)
                done += len(chunk)
                if done >= next_log:
                    log.info("  ... %d MB%s", done // (1024 * 1024),
                             f" / {total // (1024 * 1024)} MB" if total else "")
                    next_log += 100 * 1024 * 1024
    if total and tmp.stat().st_size != total:
        raise IOError(f"{dest.name}: size mismatch (got {tmp.stat().st_size}, expected {total})")
    tmp.replace(dest)


def extract_en_txt_subtree(archive: Path, country: str, out_dir: Path) -> int:
    """Extract only the `ParlaMint-{CC}-en.txt/` members; return the file count."""
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = f"ParlaMint-{country}-en.txt/"
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

    try:
        urls = discover_urls()
    except requests.RequestException as err:
        log.error("Could not reach CLARIN to discover URLs (%s). It may be temporarily "
                  "down — rerun when it's back (finished countries are skipped).", err)
        return
    if a.list:
        for cc in PARLAMINT_COUNTRIES:
            print(f"{cc:6} {urls.get(cc, '(not found)')}")
        log.info("Discovered %d/%d country archives",
                 sum(c in urls for c in PARLAMINT_COUNTRIES), len(PARLAMINT_COUNTRIES))
        return

    countries = ([c.strip() for c in a.countries.split(",") if c.strip()]
                 if a.countries else list(PARLAMINT_COUNTRIES))
    archives_dir = RAW_EN_DIR / "_archives"

    for cc in countries:
        if cc not in urls:
            log.warning("%s: no download URL found — skipping", cc)
            continue
        out_dir = RAW_EN_DIR / cc
        marker = out_dir / ".extracted"
        if marker.exists() and not a.force:
            log.info("%s: already downloaded — skipping (use --force to redo)", cc)
            continue
        archive = archives_dir / f"ParlaMint-{cc}-en.ana.tgz"
        try:
            t0 = time.time()
            if not archive.exists() or a.force:
                log.info("%s: downloading ...", cc)
                _retry(lambda: download(urls[cc], archive), f"{cc} download")
            log.info("%s: extracting English text subtree ...", cc)
            n = extract_en_txt_subtree(archive, cc, out_dir)
            marker.write_text(f"ParlaMint-en {PARLAMINT_EN_HANDLE} {cc}: {n} files, "
                              f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            if not a.keep_archives:
                archive.unlink(missing_ok=True)
            log.info("%s: done (%d files, %.1f s)", cc, n, time.time() - t0)
        except Exception as err:  # noqa: BLE001 — one bad country shouldn't kill the batch
            log.error("%s: failed after retries (%s) — continuing; rerun to resume", cc, err)


if __name__ == "__main__":
    main()
