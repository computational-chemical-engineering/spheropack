import os
import signal
import subprocess
import sys
import time

import pytest


@pytest.mark.skipif(os.name == "nt", reason="SIGINT delivery differs on Windows")
def test_ctrl_c_stops_a_long_run_quickly():
    code = (
        "import spheropack as sp\n"
        "print('started', flush=True)\n"
        "try:\n"
        "    sp.pack(n=20000, density='max', growth_rate=0.002)\n"
        "except KeyboardInterrupt:\n"
        "    print('interrupted', flush=True)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "started"
    time.sleep(1.0)
    t0 = time.monotonic()
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=10)
    assert "interrupted" in out
    assert time.monotonic() - t0 < 2.0


@pytest.mark.skipif(os.name == "nt", reason="SIGINT delivery differs on Windows")
@pytest.mark.parametrize("method", ["collisions", "event_chain"])
def test_rejection_free_runs_again_after_ctrl_c(method):
    code = (
        "import spheropack as sp\n"
        "from spheropack import rejection_free as rf\n"
        f"s = rf.System(n=2000, density=0.5, potential=rf.LennardJones(), seed=1, method={method!r})\n"
        "print('started', flush=True)\n"
        "try:\n"
        "    s.run(1e9)\n"
        "except KeyboardInterrupt:\n"
        "    print('interrupted', flush=True)\n"
        "s.run(0.01)\n"
        "print('ran again', flush=True)\n"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "started"
    time.sleep(1.0)
    proc.send_signal(signal.SIGINT)
    out, _ = proc.communicate(timeout=20)
    assert "interrupted" in out and "ran again" in out
