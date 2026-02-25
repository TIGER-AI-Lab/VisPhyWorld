(() => {
  const manifest = window.VISEXPERT_DEMO_MANIFEST;
  if (!manifest || !Array.isArray(manifest.samples) || manifest.samples.length === 0) {
    throw new Error("Missing or invalid window.VISEXPERT_DEMO_MANIFEST");
  }

  const dom = {
    prevSample: document.getElementById("prevSample"),
    nextSample: document.getElementById("nextSample"),
    sampleSelect: document.getElementById("sampleSelect"),
    sampleId: document.getElementById("sampleId"),
    sampleCounter: document.getElementById("sampleCounter"),
    promptText: document.getElementById("promptText"),
    gtFrameFirst: document.getElementById("gtFrameFirst"),
    gtFrameTenth: document.getElementById("gtFrameTenth"),
    modelBadge: document.getElementById("modelBadge"),
    modelVideo: document.getElementById("modelVideo"),
    gtVideo: document.getElementById("gtVideo"),
    modelButtons: document.getElementById("modelButtons"),
    scrollLeft: document.getElementById("scrollLeft"),
    scrollRight: document.getElementById("scrollRight"),
  };

  const models = Array.isArray(manifest.models) ? manifest.models : [];
  let sampleIndex = 0;
  let modelKey = models[0]?.key ?? (manifest.samples[0].models?.[0]?.key ?? null);

  const postHeightToParent = () => {
    try {
      if (!window.parent || window.parent === window) return;
      // Use the main layout container height instead of documentElement.scrollHeight.
      // scrollHeight is clamped by the iframe viewport height, which can prevent shrinking
      // and leave large blank space when the iframe starts taller than the content.
      const root = document.querySelector(".page") ?? document.body;
      const rect = root.getBoundingClientRect();
      const height = Math.ceil(rect.height);
      window.parent.postMessage({ type: "vpw_visexpert_demo_height", height }, window.location.origin);
    } catch {
      // Ignore cross-origin or other errors.
    }
  };

  const getModelLabel = (key) => models.find((m) => m.key === key)?.label ?? key ?? "Model";

  const setVideoSrc = (videoEl, src) => {
    if (!videoEl) return;
    const current = videoEl.getAttribute("src");
    if (current === src) return;

    try {
      videoEl.pause();
    } catch {
      // ignore
    }

    if (!src) {
      videoEl.removeAttribute("src");
      videoEl.load();
      return;
    }

    videoEl.setAttribute("src", src);
    videoEl.load();
  };

  const getSample = () => manifest.samples[sampleIndex];

  const clampIndex = (idx) => {
    const n = manifest.samples.length;
    if (n === 0) return 0;
    return ((idx % n) + n) % n;
  };

  const renderSampleSelect = () => {
    dom.sampleSelect.innerHTML = "";
    manifest.samples.forEach((s, idx) => {
      const opt = document.createElement("option");
      opt.value = String(idx);
      opt.textContent = s.label ?? s.id ?? `sample-${idx + 1}`;
      dom.sampleSelect.appendChild(opt);
    });
    dom.sampleSelect.value = String(sampleIndex);
  };

  const renderModelButtons = (sample) => {
    dom.modelButtons.innerHTML = "";

    const available = new Map();
    (sample.models ?? []).forEach((m) => {
      if (m?.key && m?.video) available.set(m.key, m);
    });

    // Ensure modelKey is valid for current sample.
    if (!available.has(modelKey)) modelKey = available.keys().next().value ?? null;

    const orderedKeys = models.length ? models.map((m) => m.key) : Array.from(available.keys());

    orderedKeys.forEach((key) => {
      const info = available.get(key);
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "model-btn";
      btn.textContent = getModelLabel(key);
      btn.disabled = !info;
      btn.setAttribute("role", "tab");
      btn.setAttribute("aria-selected", String(key === modelKey));

      btn.addEventListener("click", () => {
        if (!available.has(key)) return;
        modelKey = key;
        updateView();
      });

      dom.modelButtons.appendChild(btn);
    });
  };

  const updateView = () => {
    const sample = getSample();

    dom.sampleId.textContent = sample.label ?? sample.id ?? "—";
    dom.sampleCounter.textContent = `${sampleIndex + 1} / ${manifest.samples.length}`;
    dom.sampleSelect.value = String(sampleIndex);

    dom.promptText.textContent = sample.prompt ?? "—";

    dom.gtFrameFirst.src = sample.gt_frames?.first ?? "";
    dom.gtFrameTenth.src = sample.gt_frames?.tenth ?? "";

    setVideoSrc(dom.gtVideo, sample.gt_video ?? "");

    const byKey = new Map();
    (sample.models ?? []).forEach((m) => {
      if (m?.key) byKey.set(m.key, m);
    });
    const currentModel = byKey.get(modelKey) ?? sample.models?.[0] ?? null;

    const effectiveKey = currentModel?.key ?? modelKey;
    dom.modelBadge.textContent = getModelLabel(effectiveKey);
    setVideoSrc(dom.modelVideo, currentModel?.video ?? "");

    renderModelButtons(sample);

    // Notify the parent page so the iframe can match the demo height.
    requestAnimationFrame(() => postHeightToParent());
  };

  const changeSample = (nextIdx) => {
    sampleIndex = clampIndex(nextIdx);
    updateView();
  };

  dom.prevSample.addEventListener("click", () => changeSample(sampleIndex - 1));
  dom.nextSample.addEventListener("click", () => changeSample(sampleIndex + 1));
  dom.sampleSelect.addEventListener("change", () => changeSample(Number(dom.sampleSelect.value)));

  dom.scrollLeft.addEventListener("click", () => {
    dom.modelButtons.scrollBy({ left: -320, behavior: "smooth" });
  });
  dom.scrollRight.addEventListener("click", () => {
    dom.modelButtons.scrollBy({ left: 320, behavior: "smooth" });
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft") changeSample(sampleIndex - 1);
    if (e.key === "ArrowRight") changeSample(sampleIndex + 1);
  });

  renderSampleSelect();
  updateView();

  // Height sync for iframe embedding.
  window.addEventListener("load", postHeightToParent);
  window.addEventListener("resize", postHeightToParent);
  try {
    const ro = new ResizeObserver(() => postHeightToParent());
    ro.observe(document.body);
  } catch {
    // ignore
  }
})();
