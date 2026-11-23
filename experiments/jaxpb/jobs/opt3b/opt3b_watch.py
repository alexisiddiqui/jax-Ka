"""Local transport/polling only. Sleep between squeue polls; never run cluster numerics here."""
import datetime
import subprocess
import time
while True:
    result = subprocess.run(['python3', 'jobs/opt3b_poll.py'], text=True, capture_output=True)
    print(result.stdout, flush=True)
    if result.returncode:
        print(result.stderr, flush=True)
        raise SystemExit(result.returncode)
    active = [line for line in result.stdout.splitlines() if ' PENDING ' in line or ' RUNNING ' in line]
    if not active:
        break
    for minute in range(3):
        time.sleep(60)
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(), 'Waiting for OPT3B jobs; next squeue poll in', 2-minute, 'minutes.', flush=True)
