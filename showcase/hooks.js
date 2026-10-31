// Runs in the chat page before its own script, for the showcase: records
// everything the page plays, stands in for the microphone with a clip the
// script picks, and draws title cards and a pointer for the video.
(() => {
    const connect = AudioNode.prototype.connect;
    let master = null, mixer = null, recorder = null, saving = Promise.resolve();
    const taps = new WeakMap();

    function mixerFor(ctx) {
        if (!master) {
            master = new AudioContext({ sampleRate: 48000 });
            mixer = master.createMediaStreamDestination();
        }
        if (!ctx || ctx === master) return mixer;
        // Each of the page's audio contexts feeds the one recording through a stream of its own
        let tap = taps.get(ctx);
        if (!tap) {
            tap = ctx.createMediaStreamDestination();
            taps.set(ctx, tap);
            connect.call(master.createMediaStreamSource(tap.stream), mixer);
        }
        return tap;
    }
    // Whatever the page sends to its speakers goes to the recording too
    AudioNode.prototype.connect = function (target, ...rest) {
        const result = connect.call(this, target, ...rest);
        if (target instanceof AudioDestinationNode && !(this.context instanceof OfflineAudioContext))
            connect.call(this, mixerFor(this.context));
        return result;
    };

    function base64(blob) {
        return new Promise(resolve => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result.split(",")[1]);
            reader.readAsDataURL(blob);
        });
    }

    // The microphone: the clip set with mic(), after a moment's quiet, then silence, so voice detection ends the turn
    let clip = null;
    navigator.mediaDevices.getUserMedia = async constraints => {
        if (!constraints || !constraints.audio) throw new DOMException("No camera here", "NotFoundError");
        const ctx = new AudioContext({ sampleRate: 48000 });
        const out = ctx.createMediaStreamDestination();
        if (clip) {
            const buffer = await ctx.decodeAudioData(await (await fetch(clip)).arrayBuffer());
            const source = ctx.createBufferSource();
            source.buffer = buffer;
            connect.call(source, out);
            // The viewer hears the user speak, too
            connect.call(source, mixerFor(ctx));
            source.start(ctx.currentTime + 0.4);
            window.__showcase.micEnds = Date.now() + 400 + buffer.duration * 1000;
        }
        return out.stream;
    };

    let card = null, pointer = null;
    window.__showcase = {
        micEnds: 0,
        mic(url) { clip = url; },
        /** Starts recording the page's audio; returns when it started, in ms since the epoch. */
        record() {
            mixerFor(null);
            master.resume();
            recorder = new MediaRecorder(mixer.stream, { mimeType: "audio/webm;codecs=opus", audioBitsPerSecond: 128000 });
            recorder.ondataavailable = event => {
                if (event.data.size) saving = saving.then(async () => window.__saveAudio(await base64(event.data)));
            };
            recorder.start(5000);
            return Date.now();
        },
        /** A white flash and, in the recording only, a beep, which line the video and the sound up. */
        mark() {
            const flash = document.createElement("div");
            flash.style.cssText = "position:fixed;inset:0;z-index:100000;background:#fff";
            document.body.append(flash);
            const tone = master.createOscillator(), gain = master.createGain();
            tone.frequency.value = 1000;
            gain.gain.value = 0.5;
            connect.call(tone, gain);
            connect.call(gain, mixer);
            tone.start();
            tone.stop(master.currentTime + 0.25);
            setTimeout(() => flash.remove(), 600);
        },
        stop() {
            return new Promise(resolve => {
                recorder.onstop = () => saving.then(resolve);
                recorder.stop();
            });
        },
        card(title, subtitle) {
            card?.remove();
            card = document.createElement("div");
            card.style.cssText = "position:fixed;inset:0;z-index:99999;display:flex;flex-direction:column;align-items:center;" +
                "justify-content:center;gap:14px;background:radial-gradient(circle at 50% 40%,#1e293b,#0b0b0d 70%);" +
                "color:#f5f5f4;font:600 44px/1.2 system-ui,sans-serif;text-align:center;padding:0 10%;" +
                "opacity:0;transition:opacity .5s";
            const heading = document.createElement("div");
            heading.textContent = title;
            card.append(heading);
            if (subtitle) {
                const line = document.createElement("div");
                line.textContent = subtitle;
                line.style.cssText = "font:400 22px/1.4 system-ui,sans-serif;color:#a8a29e;max-width:900px";
                card.append(line);
            }
            document.body.append(card);
            requestAnimationFrame(() => { card.style.opacity = "1"; });
        },
        uncard() {
            const old = card;
            card = null;
            if (!old) return;
            old.style.opacity = "0";
            setTimeout(() => old.remove(), 600);
        },
        /** Moves the pointer the video shows to a point on the page. */
        point(x, y) {
            if (!pointer) {
                pointer = document.createElement("div");
                pointer.style.cssText = "position:fixed;z-index:99998;width:22px;height:22px;margin:-11px 0 0 -11px;" +
                    "border-radius:50%;background:rgba(250,250,250,.35);border:2px solid #fafafa;pointer-events:none;" +
                    "box-shadow:0 0 8px rgba(0,0,0,.6);transition:left .6s ease,top .6s ease,transform .15s;left:50%;top:60%";
                document.body.append(pointer);
            }
            pointer.style.left = `${x}px`;
            pointer.style.top = `${y}px`;
        },
        press() {
            if (!pointer) return;
            pointer.style.transform = "scale(.7)";
            setTimeout(() => { pointer.style.transform = ""; }, 180);
        },
    };
})();
