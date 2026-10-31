# Run the server on Windows from WSL

The repo lives in WSL (`~/lite-assistant`). To run the server on Windows, everything here is driven from bash in WSL: it installs the package into the global Windows Python 3.14, and starts and stops the Windows process.

[Benchmarks](../how-it-works/benchmarks.md) covers every setup tried. On the laptop these were measured on, a tuned Windows server is close to WSL but still slower, and the GPU is slower than the CPU.

The benchmarks ran on an earlier layout, so the install below hasn't been run end to end on Windows.

## Tools

- **`~/bin/python.exe`** is a WSL symlink to the Windows interpreter, `C:\Users\<you>\AppData\Local\Programs\Python\Python314\python.exe`. WSL can run Windows `.exe` files directly.
- **`powershell.exe`** runs from bash, and is used to start and stop Windows processes.
- **`U:`** is mapped to the WSL root (`\\wsl.localhost\Ubuntu`), so the WSL checkout is `U:\home\<you>\lite-assistant` to Windows.

## Install

Into the global Windows Python, with no venv. pip builds the package from the WSL checkout and installs it on `C:`, so the server doesn't read its code over the share:

```bash
cd /tmp
~/bin/python.exe -m pip install --upgrade 'U:\home\<you>\lite-assistant'
LITERTLM_BACKEND=cpu WSLENV=LITERTLM_BACKEND ~/bin/python.exe -m lite_assistant setup
```

- **`cd /tmp`:** a Windows program started from a WSL directory gets a `\\wsl.localhost\...` working directory. Some Windows tools, `cmd` among them, don't accept one.
- **Updating:** this is an install, not a link to the checkout. After changing code in WSL, run the `pip install` again.
- **`LITERTLM_BACKEND=cpu`:** setup downloads E4B (3.66 GB) only if the GPU check passes, and it passes on the Arc GPU, which is slower than the CPU here (see "GPU" below). `WSLENV` passes the variable on to the Windows process.
- **Models:** setup downloads them on the Windows side, to the Windows Hugging Face cache (`C:\Users\<you>\.cache\huggingface\hub`) and `%LOCALAPPDATA%\lite-assistant`. Downloading onto `U:` doesn't work: `huggingface_hub` can't lock or symlink files on the share.
- **Disk space:** LiteRT-LM writes compiled caches to `%LOCALAPPDATA%\lite-assistant\Cache\litertlm` on first load, about 0.9 GB for E2B on the CPU and 2.3 GB for E4B on the GPU.
- **`audioop-lts`:** installed with the package on Python 3.13 and later. It restores `audioop`, which Kokoro's resampling uses and which Python 3.13 removed.

## Start and stop the server

Stop the WSL systemd service first. WSL uses mirrored networking (`networkingMode=mirrored` in `.wslconfig`), so the two servers would compete for `localhost:5000`.

```bash
systemctl --user stop lite-assistant
```

Start the server hidden, with its output in log files:

```bash
cd /tmp && powershell.exe -NoProfile -Command "\$env:LITERTLM_BACKEND='cpu'; \$env:LITERTLM_THREADS='3'; Start-Process -FilePath 'C:\Users\<you>\AppData\Local\Programs\Python\Python314\python.exe' -ArgumentList '-u','-m','lite_assistant','serve','--port','5000' -WorkingDirectory \$env:LOCALAPPDATA -WindowStyle Hidden -RedirectStandardOutput \"\$env:LOCALAPPDATA\lite-assistant\server.out.log\" -RedirectStandardError \"\$env:LOCALAPPDATA\lite-assistant\server.err.log\""
```

- **`LITERTLM_BACKEND=cpu`:** in auto mode the engine picks the Arc GPU, because it works, but it's slower here (see "GPU" below).
- **`LITERTLM_THREADS=3`:** on Windows, decoding is fastest with 2–3 threads. LiteRT-LM's default is about 1.5× slower.
- **Power throttling:** Windows throttles a process with no visible window (EcoQoS). `lite-assistant serve` opts out at startup, and prints a warning if it can't. Without that, the server was 2–3× slower.
- **The `powershell.exe` call:** with `-PassThru`, it never returned to bash, though the server started anyway. Without `-PassThru`, it returns.
- **`-u`:** keeps Python's output unbuffered, so the log files show progress during startup.

Find or stop the server from WSL:

```bash
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object CommandLine -like '*lite_assistant*' | Select-Object ProcessId,CommandLine"
powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object CommandLine -like '*lite_assistant*' | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }"
```

Afterwards, restart the WSL service with `systemctl --user start lite-assistant`.

The tests and `scripts/benchmark.py` run from WSL against `http://localhost:5000` unchanged.

## GPU

The engine's GPU check passes on the Intel Arc integrated GPU, so with no `LITERTLM_BACKEND` the server loads E4B on it. On this laptop that's the slowest option: E4B decodes at about 2 tok/s and E2B at about 4, against 15 on the CPU. A 1,700-token prompt also hung the GPU (`DXGI_ERROR_DEVICE_HUNG`). Windows resets a GPU that is busy for about 2 seconds, and the engine needed a restart afterwards.
