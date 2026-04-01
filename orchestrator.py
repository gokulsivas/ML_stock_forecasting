import subprocess
import sys
import os
import time
import traceback
from datetime import datetime
import resend

RESEND_API_KEY = os.environ.get('RESEND_API_KEY', '')
NOTIFY_EMAIL   = os.environ.get('NOTIFY_EMAIL', '')
LOG_FILE       = 'training_logs/training_log.txt'

resend.api_key = RESEND_API_KEY


def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    os.makedirs('training_logs', exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(line + '\n')


def send_email(subject, body_html):
    if not RESEND_API_KEY or not NOTIFY_EMAIL:
        log("⚠ Resend API key or email not set. Skipping email.")
        return
    try:
        resend.Emails.send({
            "from":    "Stock Trainer <onboarding@resend.dev>",
            "to":      [NOTIFY_EMAIL],
            "subject": subject,
            "html":    body_html
        })
        log(f"✅ Email sent to {NOTIFY_EMAIL}")
    except Exception as e:
        log(f"❌ Failed to send email: {e}")


def start_postgres():
    log("Starting PostgreSQL...")
    subprocess.run(['service', 'postgresql', 'start'], check=True)
    time.sleep(3)
    log("✅ PostgreSQL started")


def restore_database():
    dump_path = 'stock_backup.dump'
    if not os.path.exists(dump_path):
        log("⚠ No dump file found at stock_backup.dump — skipping restore.")
        return

    log("Restoring database from dump...")
    result = subprocess.run([
        'pg_restore',
        '-U', 'stockuser',
        '-d', 'stockdb',
        '-v',
        dump_path
    ], capture_output=True, text=True,
       env={**os.environ, 'PGPASSWORD': 'stockpass'})

    if result.returncode == 0:
        log("✅ Database restored successfully")
    else:
        log(f"⚠ pg_restore warnings (may be harmless):\n{result.stderr[-500:]}")


def run_training():
    log("=" * 60)
    log("Starting model training...")
    log("=" * 60)

    start_time = time.time()

    process = subprocess.Popen(
        [sys.executable, '-m', 'training.train_nifty250'],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    os.makedirs('training_logs', exist_ok=True)
    with open(LOG_FILE, 'a') as logfile:
        for line in process.stdout:
            print(line, end='')
            logfile.write(line)

    process.wait()
    elapsed = time.time() - start_time
    hours   = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)

    return process.returncode, hours, minutes


def read_last_log_lines(n=30):
    try:
        with open(LOG_FILE, 'r') as f:
            lines = f.readlines()
        return ''.join(lines[-n:])
    except Exception:
        return "Could not read log file."


def main():
    log("🚀 Orchestrator started")

    # Step 1 — Start PostgreSQL
    try:
        start_postgres()
    except Exception as e:
        log(f"❌ PostgreSQL failed to start: {e}")
        send_email(
            "❌ Training Failed — PostgreSQL Error",
            f"<h2>PostgreSQL failed to start</h2><pre>{traceback.format_exc()}</pre>"
        )
        sys.exit(1)

    # Step 2 — Restore DB from dump if present
    try:
        restore_database()
    except Exception as e:
        log(f"⚠ DB restore error (continuing anyway): {e}")

    # Step 3 — Run training
    try:
        return_code, hours, minutes = run_training()
    except Exception as e:
        log(f"❌ Training crashed: {e}")
        send_email(
            "❌ Training Crashed",
            f"<h2>Training crashed with exception</h2><pre>{traceback.format_exc()}</pre>"
        )
        sys.exit(1)

    # Step 4 — Notify result
    last_logs = read_last_log_lines(30)

    if return_code == 0:
        log(f"✅ Training completed in {hours}h {minutes}m")
        send_email(
            f"✅ Training Complete — {hours}h {minutes}m",
            f"""
            <h2>✅ Stock Model Training Complete!</h2>
            <p><b>Duration:</b> {hours} hours {minutes} minutes</p>
            <p><b>Model saved to:</b> saved_models/returns_model.pth</p>
            <hr>
            <h3>Last 30 log lines:</h3>
            <pre style="background:#f4f4f4;padding:16px;border-radius:8px;font-size:13px">
{last_logs}
            </pre>
            <p style="color:gray;font-size:12px">Sent by your GPU training orchestrator</p>
            """
        )
    else:
        log(f"❌ Training failed with return code {return_code}")
        send_email(
            f"❌ Training Failed — Exit Code {return_code}",
            f"""
            <h2>❌ Training Failed</h2>
            <p><b>Exit code:</b> {return_code}</p>
            <hr>
            <h3>Last 30 log lines:</h3>
            <pre style="background:#fff0f0;padding:16px;border-radius:8px;font-size:13px">
{last_logs}
            </pre>
            """
        )
        sys.exit(1)


if __name__ == '__main__':
    main()