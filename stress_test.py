"""10-minute sustained stress test - 5 concurrent users in repeated waves."""
import time, random, threading, requests, sys
from datetime import datetime

BASE = "https://d2e6rjfv7d2rgs.cloudfront.net/coach"
API = BASE + "/api"
NUM_USERS = 5
DURATION = 660  # 10 minutes

EDITS = [(".", ". לכן, "), ("אני", "אנחנו"), ("הוא", "המומחה")]

def user_session(uid, wave, results):
    s = requests.Session()
    t = {"user": uid, "wave": wave}
    try:
        r = s.get(API + "/examples", timeout=15)
        examples = r.json()
        ex = random.choice(examples)
        r = s.get(API + "/examples/" + ex["id"], timeout=15)
        text = r.json()["text"]
        t0 = time.time()
        r = s.post(API + "/analyze", json={"text": text}, timeout=60)
        t["a1"] = int((time.time() - t0) * 1000)
        assert r.status_code == 200
        ef, et = random.choice(EDITS)
        mod = text.replace(ef, et, 1) if ef in text else text + " משפט נוסף."
        t0 = time.time()
        r = s.post(API + "/analyze", json={"text": mod}, timeout=60)
        t["a2"] = int((time.time() - t0) * 1000)
        assert r.status_code == 200
        t["ok"] = True
    except Exception as e:
        t["ok"] = False
        t["err"] = str(e)
    results.append(t)

def main():
    # Wait until 18:45 local time
    now = datetime.now()
    target = now.replace(hour=19, minute=29, second=30, microsecond=0)
    if target <= now:
        print(f"Target time 18:45 already passed (now {now.strftime('%H:%M:%S')}), starting immediately")
    else:
        wait = (target - now).total_seconds()
        print(f"Now: {now.strftime('%H:%M:%S')} | Waiting until 18:45 ({wait:.0f}s)...")
        time.sleep(wait)

    print(f"\nStarted at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Stress: {NUM_USERS} users for {DURATION}s")
    print("=" * 60)

    all_r = []
    wave = 0
    start = time.time()
    while time.time() - start < DURATION:
        wave += 1
        res = []
        ts = [threading.Thread(target=user_session, args=(i+1, wave, res)) for i in range(NUM_USERS)]
        for t in ts:
            t.start()
            time.sleep(0.2)
        for t in ts:
            t.join(120)
        all_r.extend(res)
        ok = [r for r in res if r.get("ok")]
        if ok:
            avg = sum(r["a1"] for r in ok) // len(ok)
            mx = max(r["a1"] for r in ok)
            print(f"W{wave:3d}: {len(ok)}/{NUM_USERS} OK avg={avg}ms max={mx}ms t={time.time()-start:.0f}s")
        for r in res:
            if not r.get("ok"):
                print(f"W{wave:3d}: U{r['user']} FAIL {r.get('err','')[:60]}")
        time.sleep(1)

    ok_all = [r for r in all_r if r.get("ok")]
    fail_all = [r for r in all_r if not r.get("ok")]
    print(f"\n{'='*60}")
    print(f"Finished at {datetime.now().strftime('%H:%M:%S')}")
    print(f"Duration: {time.time()-start:.0f}s | Waves: {wave} | Total: {len(all_r)}")
    print(f"OK: {len(ok_all)} | FAIL: {len(fail_all)} | Rate: {len(ok_all)/max(len(all_r),1)*100:.1f}%")
    if ok_all:
        a1 = sorted(r["a1"] for r in ok_all)
        a2 = sorted(r["a2"] for r in ok_all)
        print(f"Analyze:    avg={sum(a1)//len(a1)}ms p50={a1[len(a1)//2]}ms p95={a1[int(len(a1)*0.95)]}ms max={a1[-1]}ms")
        print(f"Re-analyze: avg={sum(a2)//len(a2)}ms p50={a2[len(a2)//2]}ms p95={a2[int(len(a2)*0.95)]}ms max={a2[-1]}ms")
    if fail_all:
        print(f"\nFailures ({len(fail_all)}):")
        for r in fail_all[:20]:
            print(f"  Wave {r['wave']} User {r['user']}: {r.get('err','')[:80]}")

if __name__ == "__main__":
    main()
