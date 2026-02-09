const TARGET_FPS = 5;
// No forced global minimum duration; prefer caller-requested duration, else use injected target.
const DEFAULT_MIN_DURATION_MS = 0;

function _codexNum(value, fallback) {
    const v = Number(value);
    return Number.isFinite(v) ? v : fallback;
}

function _codexGetTargetFps() {
    try {
        const v = _codexNum(window.__codexTargetFps, NaN);
        return v > 0 ? v : TARGET_FPS;
    } catch (e) {
        return TARGET_FPS;
    }
}

function _codexGetTargetDurationMs() {
    try {
        const v = _codexNum(window.__codexTargetDurationMs ?? window.__codexMinRecordingDurationMs, NaN);
        return v > 0 ? v : null;
    } catch (e) {
        return null;
    }
}

function setupRecording(canvas, maxDuration = DEFAULT_MIN_DURATION_MS) {
    // Normalize input: accept an HTMLCanvasElement or common wrappers (e.g. p5.Renderer).
    try {
        if (canvas && typeof canvas.captureStream !== 'function') {
            if (canvas.elt && typeof canvas.elt.captureStream === 'function') {
                canvas = canvas.elt;
            } else if (canvas.canvas && typeof canvas.canvas.captureStream === 'function') {
                canvas = canvas.canvas;
            }
        }
    } catch (e) {
        // ignore
    }
    // Choose duration (best-effort):
    // - Respect caller-requested duration if provided (>0).
    // - Otherwise, fall back to injected target duration (e.g., original video duration).
    // - Finally, fall back to 10s to avoid producing a 0-length video.
    const injectedTarget = _codexGetTargetDurationMs(); // may be null
    const requestedDuration = _codexNum(maxDuration, NaN);
    const requested = (Number.isFinite(requestedDuration) && requestedDuration > 0) ? requestedDuration : 0;
    if (requested > 0) {
        maxDuration = requested;
    } else if (injectedTarget) {
        maxDuration = injectedTarget;
    } else {
        maxDuration = DEFAULT_MIN_DURATION_MS;
    }
    if (!maxDuration || maxDuration <= 0) {
        maxDuration = 10000;
    }

    try {
        if (window.__codexRecordingActive) {
            // If a recording is already running, try to extend its stop timer when asked for longer.
            const recorder = window.__codexActiveRecorder || null;
            if (recorder && recorder.state === 'recording') {
                const planned = _codexNum(window.__codexPlannedStopMs, 0);
                if (maxDuration > planned) {
                    window.__codexPlannedStopMs = maxDuration;
                    try {
                        if (window.__codexStopTimer) clearTimeout(window.__codexStopTimer);
                    } catch (e) {}
                    const startTs = _codexNum(window.__codexRecordingStartTs, performance.now());
                    const elapsed = performance.now() - startTs;
                    const remaining = Math.max(0, maxDuration - elapsed);
                    window.__codexStopTimer = setTimeout(() => {
                        try {
                            if (recorder.state === 'recording') {
                                console.log(`⏰ ${maxDuration/1000}s reached, stopping recording...`);
                                recorder.stop();
                            }
                        } catch (e) {}
                    }, remaining);
                    console.log(`⏱️ Extended recording to ${maxDuration/1000}s`);
                }
                return recorder;
            }
            console.warn('⚠️ Recording already active; skipping duplicate setupRecording call');
            return recorder;
        }
        window.__codexRecordingActive = true;
    } catch (e) {
        // ignore
    }
    const targetFps = _codexGetTargetFps();
    console.log(`🎥 Setting up recording at ${targetFps} fps with ${maxDuration/1000}s duration limit`);
    
    // Start capturing the stream
    const stream = canvas.captureStream(targetFps); // Capture at dataset FPS

    // Set up the MediaRecorder
    const recordedChunks = [];
    const mediaRecorder = new MediaRecorder(stream, { mimeType: 'video/webm' });

    // Listen for dataavailable to store video chunks
    mediaRecorder.ondataavailable = event => {
        if (event.data.size > 0) {
            recordedChunks.push(event.data);
        }
    };

    // When the recording is stopped, create a download link
    mediaRecorder.onstop = () => {
        console.log('🎬 Recording stopped, creating download...');
        const blob = new Blob(recordedChunks, { type: 'video/webm' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.style.display = 'none';
        a.href = url;
        a.download = `output.webm`;
        document.body.appendChild(a);
        a.click();
        URL.revokeObjectURL(url);
        console.log('✅ Video download initiated');
    };

    // Auto-start recording immediately
    mediaRecorder.start();
    console.log('🔴 Recording started');
    try {
        window.__codexActiveRecorder = mediaRecorder;
        window.__codexRecordingStartTs = performance.now();
        window.__codexPlannedStopMs = maxDuration;
    } catch (e) {
        // ignore
    }
    
    // Auto-stop after maxDuration milliseconds
    const stopTimer = setTimeout(() => {
        if (mediaRecorder.state === 'recording') {
            console.log(`⏰ ${maxDuration/1000}s reached, stopping recording...`);
            mediaRecorder.stop();
        }
    }, maxDuration);
    try {
        window.__codexStopTimer = stopTimer;
    } catch (e) {
        // ignore
    }

    return mediaRecorder;
}
