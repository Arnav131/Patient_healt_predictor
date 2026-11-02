/**
 * CardioPulse AI - Interactive Application Engine
 * Handles three-way theme switching, biometrics sliders, live BMI/PP badges,
 * model inference, report parsing, and dynamic SVG circular risk gauge.
 */

document.addEventListener("DOMContentLoaded", () => {
    initThemeSwitcher();
    initSynchronizedSliders();
    initTierSelectors();
    initModeTabs();
    initFormSubmission();
    initFileUpload();
    initPresetCards();
    initScrollReveal();
});

/* ==========================================================================
   Toast Notification System
   ========================================================================== */
let toastTimeout = null;
function showToast(message) {
    const toast = document.getElementById("toastNotification");
    if (!toast) return;
    toast.textContent = message;
    toast.classList.add("show");
    if (toastTimeout) clearTimeout(toastTimeout);
    toastTimeout = setTimeout(() => {
        toast.classList.remove("show");
    }, 3200);
}

/* ==========================================================================
   1. Theme Switcher (Light Default, Warm Linen, Slate Dark)
   ========================================================================== */
function initThemeSwitcher() {
    const savedTheme = localStorage.getItem("cardiopulse_theme") || "light";
    setTheme(savedTheme);

    document.querySelectorAll(".theme-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            const theme = btn.getAttribute("data-theme-choice");
            setTheme(theme);
        });
    });
}

function setTheme(theme) {
    if (theme === "light") {
        document.documentElement.removeAttribute("data-theme");
    } else {
        document.documentElement.setAttribute("data-theme", theme);
    }
    localStorage.setItem("cardiopulse_theme", theme);

    document.querySelectorAll(".theme-btn").forEach(btn => {
        btn.classList.toggle("active", btn.getAttribute("data-theme-choice") === theme);
    });
}

/* ==========================================================================
   2. Synchronized Sliders, Number Inputs & Real-Time Biomarker Badges
   ========================================================================== */
function initSynchronizedSliders() {
    const fields = [
        { range: "ageRange", num: "ageNum" },
        { range: "heightRange", num: "heightNum" },
        { range: "weightRange", num: "weightNum" },
        { range: "apHiRange", num: "apHiNum" },
        { range: "apLoRange", num: "apLoNum" }
    ];

    fields.forEach(({ range, num }) => {
        const rEl = document.getElementById(range);
        const nEl = document.getElementById(num);
        if (!rEl || !nEl) return;

        rEl.addEventListener("input", () => {
            nEl.value = rEl.value;
            updateLiveCalculations();
        });

        nEl.addEventListener("input", () => {
            rEl.value = nEl.value;
            updateLiveCalculations();
        });
    });

    updateLiveCalculations();
}

function updateLiveCalculations() {
    const height = parseFloat(document.getElementById("heightNum")?.value || 170);
    const weight = parseFloat(document.getElementById("weightNum")?.value || 70);
    const apHi = parseFloat(document.getElementById("apHiNum")?.value || 120);
    const apLo = parseFloat(document.getElementById("apLoNum")?.value || 80);

    // 1. Live BMI Badge
    const heightM = height / 100.0;
    const bmi = (weight / (heightM * heightM)).toFixed(1);
    let bmiCat = "Normal";
    if (bmi < 18.5) bmiCat = "Underweight";
    else if (bmi < 25.0) bmiCat = "Normal";
    else if (bmi < 30.0) bmiCat = "Overweight";
    else bmiCat = "Obese";

    const bmiBadge = document.getElementById("liveBmiBadge");
    if (bmiBadge) {
        bmiBadge.textContent = `BMI: ${bmi} (${bmiCat})`;
    }

    // 2. Live Pulse Pressure & BP Staging
    const pp = Math.round(apHi - apLo);
    let bpStage = "Normal BP";
    if (apHi > 180 || apLo > 120) bpStage = "Crisis";
    else if (apHi >= 140 || apLo >= 90) bpStage = "Stage 2";
    else if (apHi >= 130 || apLo >= 80) bpStage = "Stage 1";
    else if (apHi >= 120 && apLo < 80) bpStage = "Elevated";

    const bpBadge = document.getElementById("liveBpBadge");
    if (bpBadge) {
        bpBadge.textContent = `PP: ${pp} mmHg | ${bpStage}`;
    }
}

/* ==========================================================================
   3. Tier Selectors (Cholesterol & Glucose)
   ========================================================================== */
function initTierSelectors() {
    setupTierGroup("cholesterolOptions", "cholesterolInput");
    setupTierGroup("glucOptions", "glucInput");
}

function setupTierGroup(containerId, hiddenInputId) {
    const container = document.getElementById(containerId);
    const hidden = document.getElementById(hiddenInputId);
    if (!container || !hidden) return;

    container.querySelectorAll(".tier-option").forEach(opt => {
        opt.addEventListener("click", () => {
            container.querySelectorAll(".tier-option").forEach(o => o.classList.remove("selected"));
            opt.classList.add("selected");
            hidden.value = opt.getAttribute("data-value");
        });
    });
}

/* ==========================================================================
   4. Mode Switching Tabs (Direct / Report Upload / Presets)
   ========================================================================== */
function initModeTabs() {
    const tabs = document.querySelectorAll(".mode-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            tabs.forEach(t => t.classList.remove("active"));
            tab.classList.add("active");

            const mode = tab.getAttribute("data-mode");
            document.getElementById("manualInputPane").style.display = mode === "manual" ? "block" : "none";
            document.getElementById("uploadInputPane").style.display = mode === "upload" ? "block" : "none";
        });
    });
}

/* ==========================================================================
   5. Form Submission & Risk Prediction Execution
   ========================================================================== */
function initFormSubmission() {
    const btn = document.getElementById("btnAnalyzeRisk");
    if (btn) {
        btn.addEventListener("click", () => executePrediction(btn));
    }

    const resetManualBtn = document.getElementById("btnResetPatientManual");
    if (resetManualBtn) {
        resetManualBtn.addEventListener("click", resetPatientConsole);
    }

    const resetTopBtn = document.getElementById("btnResetConsole");
    if (resetTopBtn) {
        resetTopBtn.addEventListener("click", resetPatientConsole);
    }

    const btnFineTune = document.getElementById("btnFineTuneBiometrics");
    if (btnFineTune) {
        btnFineTune.addEventListener("click", switchToManualTab);
    }

    const btnAnalyzeUpload = document.getElementById("btnAnalyzeFromUpload");
    if (btnAnalyzeUpload) {
        btnAnalyzeUpload.addEventListener("click", () => executePrediction(btnAnalyzeUpload));
    }

    const printBtn = document.getElementById("btnPrintReport");
    if (printBtn) {
        printBtn.addEventListener("click", () => window.print());
    }
}

function switchToManualTab() {
    const tabs = document.querySelectorAll(".mode-tab");
    tabs.forEach(t => t.classList.toggle("active", t.getAttribute("data-mode") === "manual"));
    document.getElementById("manualInputPane").style.display = "block";
    document.getElementById("uploadInputPane").style.display = "none";
    showToast("Switched to Direct Patient Biometrics for fine-tuning");
    document.getElementById("diagnosticConsole").scrollIntoView({ behavior: "smooth" });
}

async function executePrediction(triggerBtn = null) {
    const mainBtn = document.getElementById("btnAnalyzeRisk");
    const uploadBtn = document.getElementById("btnAnalyzeFromUpload");
    const activeBtn = triggerBtn || mainBtn;
    const originalText = activeBtn ? activeBtn.innerHTML : "";

    if (activeBtn) {
        activeBtn.innerHTML = `<span class="spinner"></span> Computing Hemodynamics...`;
        activeBtn.disabled = true;
    }
    if (mainBtn && mainBtn !== activeBtn) mainBtn.disabled = true;
    if (uploadBtn && uploadBtn !== activeBtn) uploadBtn.disabled = true;

    try {
        const payload = {
            age: parseFloat(document.getElementById("ageNum").value),
            gender: parseInt(document.getElementById("genderSelect").value),
            height: parseFloat(document.getElementById("heightNum").value),
            weight: parseFloat(document.getElementById("weightNum").value),
            ap_hi: parseFloat(document.getElementById("apHiNum").value),
            ap_lo: parseFloat(document.getElementById("apLoNum").value),
            cholesterol: parseInt(document.getElementById("cholesterolInput").value),
            gluc: parseInt(document.getElementById("glucInput").value),
            smoke: document.getElementById("smokeToggle").checked ? 1 : 0,
            alco: document.getElementById("alcoToggle").checked ? 1 : 0,
            active: document.getElementById("activeToggle").checked ? 1 : 0
        };

        const resp = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || `Server error (${resp.status})`);
        }
        const data = await resp.json();

        renderResults(data);
    } catch (err) {
        showToast("Prediction Error: " + err.message);
    } finally {
        if (activeBtn) {
            activeBtn.innerHTML = originalText;
            activeBtn.disabled = false;
        }
        if (mainBtn) mainBtn.disabled = false;
        if (uploadBtn) uploadBtn.disabled = false;
    }
}

function renderResults(data) {
    document.getElementById("resultsEmptyState").style.display = "none";
    const activeCard = document.getElementById("resultsActiveCard");
    activeCard.style.display = "flex";

    // 1. Animate SVG Circular Dial
    const pct = data.risk_percentage;
    const circle = document.getElementById("riskProgressCircle");
    const pctLabel = document.getElementById("riskPctLabel");
    const tierPill = document.getElementById("riskTierPill");

    const radius = 80;
    const circumference = 2 * Math.PI * radius; // ~502.65
    circle.style.strokeDasharray = `${circumference} ${circumference}`;

    const offset = circumference - (pct / 100) * circumference;
    circle.style.strokeDashoffset = offset;

    // Color coordination based on risk
    let strokeColor = "#10b981"; // Emerald
    let tierClass = "low";
    if (pct >= 75) {
        strokeColor = "#e11d48"; // Crimson
        tierClass = "critical";
    } else if (pct >= 50) {
        strokeColor = "#f97316"; // Orange
        tierClass = "high";
    } else if (pct >= 25) {
        strokeColor = "#f59e0b"; // Amber
        tierClass = "moderate";
    }

    circle.style.stroke = strokeColor;
    pctLabel.textContent = `${pct}%`;
    tierPill.className = `risk-tier-pill ${tierClass}`;
    tierPill.textContent = data.risk_tier;

    // 2. Populate Biometric Cards
    document.getElementById("resBmi").textContent = `${data.biomarkers.bmi} kg/m²`;
    document.getElementById("resBmiSub").textContent = data.biomarkers.bmi_category;

    document.getElementById("resPp").textContent = `${data.biomarkers.pulse_pressure} mmHg`;
    document.getElementById("resMap").textContent = `MAP: ${data.biomarkers.mean_arterial_pressure} mmHg`;

    document.getElementById("resBpStage").textContent = data.biomarkers.aha_bp_stage;
    document.getElementById("resBpSub").textContent = data.biomarkers.aha_description;

    document.getElementById("resMetabolic").textContent = `${data.biomarkers.metabolic_score} / 4`;
    document.getElementById("resMetabolicSub").textContent = "Risk Multipliers";

    // 3. Clinical Protocol
    document.getElementById("resRecommendation").textContent = data.recommendation;
}

/* ==========================================================================
   6. Lab Report & Prescription File Upload / OCR
   ========================================================================== */
function initFileUpload() {
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("reportFileInput");
    if (!dropZone || !fileInput) return;

    dropZone.addEventListener("click", () => fileInput.click());

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length > 0) {
            handleReportUpload(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            handleReportUpload(fileInput.files[0]);
        }
    });
}

async function handleReportUpload(file) {
    const statusBox = document.getElementById("uploadStatusText");
    statusBox.textContent = `Processing and extracting: ${file.name}...`;

    // Reset right column so previous patient results do not linger
    const activeCard = document.getElementById("resultsActiveCard");
    const emptyState = document.getElementById("resultsEmptyState");
    if (activeCard) activeCard.style.display = "none";
    if (emptyState) {
        emptyState.style.display = "block";
        const h3 = emptyState.querySelector("h3");
        const p = emptyState.querySelector("p");
        if (h3) h3.textContent = "Report Extracted: Awaiting Analysis";
        if (p) p.textContent = "Clinical biomarkers parsed from document. Review indicators below, fine-tune, or click 'Start Analysing Risk' to evaluate.";
    }

    const formData = new FormData();
    formData.append("file", file);

    try {
        const resp = await fetch("/api/extract-report", {
            method: "POST",
            body: formData
        });

        if (!resp.ok) {
            const errData = await resp.json().catch(() => ({}));
            throw new Error(errData.detail || `Upload extraction failed (${resp.status})`);
        }
        const result = await resp.json();

        // Populate review drawer
        const drawer = document.getElementById("reviewDrawer");
        const list = document.getElementById("extractedItemsList");
        list.innerHTML = "";

        if (result.detected_count === 0) {
            list.innerHTML = `<li>No clinical biomarkers detected in document text. Please enter values manually.</li>`;
        } else {
            result.evidence.forEach(item => {
                const li = document.createElement("li");
                li.textContent = `✓ ${item}`;
                li.style.color = "var(--success-green)";
                li.style.fontWeight = "600";
                list.appendChild(li);
            });

            // Auto-populate manual inputs
            const d = result.extracted_data;
            if (d.age) setVal("ageRange", "ageNum", d.age);
            if (d.gender) document.getElementById("genderSelect").value = d.gender;
            if (d.height) setVal("heightRange", "heightNum", d.height);
            if (d.weight) setVal("weightRange", "weightNum", d.weight);
            if (d.ap_hi) setVal("apHiRange", "apHiNum", d.ap_hi);
            if (d.ap_lo) setVal("apLoRange", "apLoNum", d.ap_lo);
            if (d.cholesterol) selectTier("cholesterolOptions", "cholesterolInput", d.cholesterol);
            if (d.gluc) selectTier("glucOptions", "glucInput", d.gluc);
            if (d.smoke !== undefined) document.getElementById("smokeToggle").checked = (d.smoke === 1);

            updateLiveCalculations();
        }

        drawer.style.display = "block";
        statusBox.textContent = `Extraction complete! (${result.detected_count} biomarkers identified)`;
        showToast(`Extracted ${result.detected_count} biomarkers from lab report`);
    } catch (err) {
        statusBox.textContent = "Error: " + err.message;
        showToast("Extraction error: " + err.message);
    }
}

function resetPatientConsole() {
    // 1. Reset inputs to baseline defaults
    setVal("ageRange", "ageNum", 50);
    const genderSel = document.getElementById("genderSelect");
    if (genderSel) genderSel.value = "2";
    setVal("heightRange", "heightNum", 172);
    setVal("weightRange", "weightNum", 70);
    setVal("apHiRange", "apHiNum", 120);
    setVal("apLoRange", "apLoNum", 80);
    selectTier("cholesterolOptions", "cholesterolInput", 1);
    selectTier("glucOptions", "glucInput", 1);

    const smokeToggle = document.getElementById("smokeToggle");
    if (smokeToggle) smokeToggle.checked = false;
    const alcoToggle = document.getElementById("alcoToggle");
    if (alcoToggle) alcoToggle.checked = false;
    const activeToggle = document.getElementById("activeToggle");
    if (activeToggle) activeToggle.checked = true;

    // 2. Clear Upload & Review Drawer
    const fileInput = document.getElementById("reportFileInput");
    if (fileInput) fileInput.value = "";
    const uploadStatus = document.getElementById("uploadStatusText");
    if (uploadStatus) uploadStatus.textContent = "";
    const reviewDrawer = document.getElementById("reviewDrawer");
    if (reviewDrawer) reviewDrawer.style.display = "none";
    const extractedList = document.getElementById("extractedItemsList");
    if (extractedList) extractedList.innerHTML = "";

    // 3. Reset Results Pane on Right Column
    const activeCard = document.getElementById("resultsActiveCard");
    const emptyState = document.getElementById("resultsEmptyState");
    if (activeCard) activeCard.style.display = "none";
    if (emptyState) {
        emptyState.style.display = "block";
        const h3 = emptyState.querySelector("h3");
        const p = emptyState.querySelector("p");
        if (h3) h3.textContent = "Awaiting Patient Ingestion";
        if (p) p.textContent = "Adjust biometric parameters or upload a lab report, then click \"Analyze\" to execute model inference.";
    }

    // 4. Reset Dial offset
    const circle = document.getElementById("riskProgressCircle");
    if (circle) circle.style.strokeDashoffset = "502.65";

    // 5. Reset live calculations & badges
    updateLiveCalculations();

    // 6. Tactile feedback
    showToast("Patient console reset for new intake");
}

function setVal(rangeId, numId, val) {
    const r = document.getElementById(rangeId);
    const n = document.getElementById(numId);
    if (r) r.value = val;
    if (n) n.value = val;
}

function selectTier(containerId, hiddenId, value) {
    const container = document.getElementById(containerId);
    const hidden = document.getElementById(hiddenId);
    if (!container || !hidden) return;
    hidden.value = value;
    container.querySelectorAll(".tier-option").forEach(opt => {
        opt.classList.toggle("selected", opt.getAttribute("data-value") == value);
    });
}

/* ==========================================================================
   7. Clinical Presets Handler
   ========================================================================== */
function initPresetCards() {
    document.querySelectorAll(".btn-load-preset").forEach(btn => {
        btn.addEventListener("click", () => {
            const presetKey = btn.getAttribute("data-preset");
            loadPreset(presetKey);
        });
    });
}

function loadPreset(key) {
    const presets = {
        healthy: { age: 28, gender: 1, height: 165, weight: 58, ap_hi: 116, ap_lo: 76, cholesterol: 1, gluc: 1, smoke: false, alco: false, active: true },
        prehypertensive: { age: 45, gender: 2, height: 174, weight: 82, ap_hi: 134, ap_lo: 86, cholesterol: 2, gluc: 1, smoke: false, alco: true, active: false },
        critical: { age: 58, gender: 2, height: 172, weight: 96, ap_hi: 152, ap_lo: 96, cholesterol: 3, gluc: 2, smoke: true, alco: false, active: false }
    };

    const p = presets[key];
    if (!p) return;

    // Switch to manual tab
    document.querySelectorAll(".mode-tab").forEach(t => t.classList.toggle("active", t.getAttribute("data-mode") === "manual"));
    document.getElementById("manualInputPane").style.display = "block";
    document.getElementById("uploadInputPane").style.display = "none";

    setVal("ageRange", "ageNum", p.age);
    document.getElementById("genderSelect").value = p.gender;
    setVal("heightRange", "heightNum", p.height);
    setVal("weightRange", "weightNum", p.weight);
    setVal("apHiRange", "apHiNum", p.ap_hi);
    setVal("apLoRange", "apLoNum", p.ap_lo);
    selectTier("cholesterolOptions", "cholesterolInput", p.cholesterol);
    selectTier("glucOptions", "glucInput", p.gluc);
    document.getElementById("smokeToggle").checked = p.smoke;
    document.getElementById("alcoToggle").checked = p.alco;
    document.getElementById("activeToggle").checked = p.active;

    updateLiveCalculations();
    executePrediction();

    // Smooth scroll up to console
    document.getElementById("diagnosticConsole").scrollIntoView({ behavior: "smooth" });
}

/* ==========================================================================
   8. Scroll Reveal Observer
   ========================================================================== */
function initScrollReveal() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add("revealed");
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll(".reveal-on-scroll").forEach(el => observer.observe(el));
}
