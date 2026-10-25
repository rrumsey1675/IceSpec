#!/usr/bin/env python3

# batch_runner.py
from __future__ import annotations
import csv
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

# =========[ Config ]=========
SCRIPT_TO_RUN   = "spectralBubbles.py"                   # script to call
SCRIPT_ID       = "spectralBubbles_v3.3"              # script version label
BATCH_ID        = os.getenv("BATCH_ID", "DIC1-Bubbles")
BATCH_TIMESTAMP = os.getenv("BATCH_TIMESTAMP", datetime.now().strftime("%Y-%m-%d_%H%M%S"))
TIMEOUT_SEC     = 60 * 90                             # 1.5 hour per job

# image IDs here, or load from a file (see TARGETS loader below)
TARGETS = []

# If a ./targets.txt exists; overrides TARGETS above.
TARGETS_FILE = Path("targets-DIC.txt")

# Where to store batch logs
LOG_DIR   = Path("/data/icecores/Output/batch_logs") / f"{BATCH_ID}_{SCRIPT_ID}_{BATCH_TIMESTAMP}"
CSV_LOG   = LOG_DIR / f"batch_log{BATCH_TIMESTAMP}.csv"
JSONL_LOG = LOG_DIR / f"batch_log{BATCH_TIMESTAMP}.jsonl"
LOG_DIR.mkdir(parents=True, exist_ok=True)
# ============================


# =========[ Helpers ]========
def load_targets() -> list[str]:
    if TARGETS_FILE.exists():
        return [t.strip() for t in TARGETS_FILE.read_text().splitlines()
                if t.strip() and not t.strip().startswith("#")]
    return TARGETS

def write_csv_header(path: Path) -> None:
    if not path.exists():
        with path.open("w", newline="") as f:
            csv.writer(f).writerow([
                "ts_utc","script_id","batch_id","batch_timestamp",
                "image_id","returncode","duration_s","output_dir","readme_path","stderr_excerpt"
            ])

def append_csv(path: Path, row: list) -> None:
    with path.open("a", newline="") as f:
        csv.writer(f).writerow(row)

def append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(obj) + "\n")

def excerpt(s: str | None, n: int = 1400) -> str:
    s = (s or "").strip().replace("\n", " ")
    return s[:n]

def sniff_stdout(stdout: str) -> tuple[str, str]:
    """Look for lines like 'output_dir: ...' and 'readme: ...' from the child script."""
    out_dir, readme_path = "", ""
    for line in stdout.splitlines():
        low = line.lower()
        if low.startswith("output_dir:"):
            out_dir = line.split(":", 1)[1].strip()
        elif low.startswith("readme:"):
            readme_path = line.split(":", 1)[1].strip()
    return out_dir, readme_path
# ============================

# =========[ Alert ]========

def notify_email_curl(subject: str, body: str) -> None:
    """Send an email using curl + SMTP. Requires env vars for credentials."""
    smtp_url  = os.getenv("SMTP_URL", "smtp://smtp.gmail.com:587")
    user      = os.getenv("SMTP_USER")
    password  = os.getenv("SMTP_PASS")
    to_addr   = os.getenv("ALERT_EMAIL_TO")
    from_addr = user

    if not all([user, password, to_addr]):
        print("[notify] Email env vars not set, skipping email")
        return

    # Build the raw email message
    email_data = (
        f"From: {from_addr}\n"
        f"To: {to_addr}\n"
        f"Subject: {subject}\n"
        f"Content-Type: text/plain; charset=UTF-8\n\n"
        f"{body}\n"
    )

    try:
        cp = subprocess.run(
            [
                "curl",
                "--url", smtp_url,
                "--ssl-reqd",
                "--mail-from", from_addr,
                "--mail-rcpt", to_addr,
                "--user", f"{user}:{password}",
                "-T", "-"  # read message from stdin
            ],
            input=email_data,
            text=True,
            capture_output=True,
            check=False,
        )
        if cp.returncode == 0:
            print("[notify] Email sent successfully via curl")
        else:
            print(f"[notify] Email failed (code={cp.returncode}): {cp.stderr.strip()}")
    except Exception as e:
        print(f"[notify] Could not send email: {e}")

# ============================


# =========[ Main ]===========
def main() -> None:
    targets = load_targets()
    write_csv_header(CSV_LOG)

    # Env passed to children (lets them tag README + optionally unify timestamps)
    base_env = os.environ.copy()
    base_env["BATCH_ID"]        = BATCH_ID
    base_env["BATCH_TIMESTAMP"] = BATCH_TIMESTAMP
    base_env["SCRIPT_ID"]       = SCRIPT_ID

    print(f"[batch] SCRIPT={SCRIPT_TO_RUN}  SCRIPT_ID={SCRIPT_ID}  BATCH_ID={BATCH_ID}  TS={BATCH_TIMESTAMP}")
    print(f"[batch] Targets: {len(targets)} items")

    for image_id in targets:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        cmd = ["python", SCRIPT_TO_RUN, image_id, BATCH_ID]
        start = time.time()

        print(f"[batch] start {image_id}")
        try:
            cp = subprocess.run(
                cmd,
                env=base_env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SEC,
                check=False
            )
            rc, out, err = cp.returncode, (cp.stdout or ""), (cp.stderr or "")
        except subprocess.TimeoutExpired as e:
            rc, out, err = 124, "", f"TimeoutExpired: {e}"
        except Exception as e:
            rc, out, err = 1, "", f"UnexpectedError: {e!r}"

        dur = round(time.time() - start, 3)
        output_dir, readme_path = sniff_stdout(out)

        append_csv(CSV_LOG, [
            ts, SCRIPT_ID, BATCH_ID, BATCH_TIMESTAMP,
            image_id, rc, dur, output_dir, readme_path, excerpt(err)
        ])
        append_jsonl(JSONL_LOG, {
            "ts_utc": ts, "script_id": SCRIPT_ID, "batch_id": BATCH_ID, "batch_timestamp": BATCH_TIMESTAMP,
            "image_id": image_id, "returncode": rc, "duration_s": dur,
            "output_dir": output_dir, "readme_path": readme_path,
            "stderr_excerpt": excerpt(err), "stdout_excerpt": excerpt(out)
        })

        # terminal output
        if rc == 0:
            print(f"[batch] OK {image_id} in {dur}s")
        else:
            print(f"[batch] FAIL({rc}) {image_id} in {dur}s")
            if err.strip():
                print("------ STDERR BEGIN ------")
                print(err.strip())
                print("------- STDERR END -------")

    print(f"[batch] logs:\n  {CSV_LOG}\n  {JSONL_LOG}")
    summary = f"""SCRIPT_ID={SCRIPT_ID}
    BATCH_ID={BATCH_ID}
    TIMESTAMP={BATCH_TIMESTAMP}
    CSV: {CSV_LOG}
    JSONL: {JSONL_LOG}
    """
    notify_email_curl(f"Batch complete: {BATCH_ID}", summary)


# ============================
if __name__ == "__main__":
    main()

