"use strict";

(() => {
    const shell = document.querySelector(".editor-shell");
    const form = document.getElementById("manualEditorForm");

    if (!shell || !form) {
        return;
    }

    const previewUrl = shell.dataset.previewUrl;
    const csrfToken = shell.dataset.csrfToken;
    const previewImage = document.getElementById("editorPreview");
    const previewLoader = document.getElementById("previewLoader");
    const previewStatus = document.getElementById("previewStatus");
    const finalizeButton = document.getElementById("finalizeEditor");
    const contradictionGuard = document.getElementById("contradictionGuard");
    const contradictionReason = document.getElementById("contradictionReason");
    const tabs = Array.from(document.querySelectorAll(".category-tab"));
    const sections = Array.from(document.querySelectorAll(".component-section"));
    const radios = Array.from(form.querySelectorAll('input[type="radio"]'));

    let requestInFlight = false;
    let updatePending = false;
    let debounceTimer = null;

    function setStatus(message, type = "") {
        if (!previewStatus) {
            return;
        }

        previewStatus.textContent = message;
        previewStatus.classList.toggle("is-error", type === "error");
        previewStatus.classList.toggle("is-success", type === "success");
    }

    function setLoading(isLoading) {
        if (previewLoader) {
            previewLoader.hidden = !isLoading;
        }
    }

    function markSelectedCards() {
        document.querySelectorAll(".component-card").forEach((card) => {
            const input = card.querySelector('input[type="radio"]');
            card.classList.toggle("is-selected", Boolean(input && input.checked));
        });
    }

    function updateContradictionGuard() {
        const contradictorySelections = radios.filter((radio) => {
            const card = radio.closest(".component-card");
            return radio.checked && card && card.classList.contains("status-contradictory");
        });
        const hasContradiction = contradictorySelections.length > 0;

        if (contradictionGuard) {
            contradictionGuard.hidden = !hasContradiction;
        }
        if (contradictionReason) {
            contradictionReason.required = hasContradiction;
            if (!hasContradiction) {
                contradictionReason.setCustomValidity("");
            }
        }
        return hasContradiction;
    }

    function collectChoices() {
        const choices = {};
        new FormData(form).forEach((value, key) => {
            choices[key] = value;
        });
        return choices;
    }

    function openCategory(targetId, focusTab = false) {
        tabs.forEach((tab) => {
            const active = tab.dataset.categoryTarget === targetId;
            tab.classList.toggle("active", active);
            tab.setAttribute("aria-selected", String(active));
            tab.setAttribute("tabindex", active ? "0" : "-1");
            if (active && focusTab) {
                tab.focus();
            }
        });

        sections.forEach((section) => {
            section.classList.toggle("active", section.id === targetId);
        });
    }

    async function sendPreviewUpdate() {
        if (!previewUrl) {
            setStatus("Preview endpoint is unavailable.", "error");
            return;
        }

        if (requestInFlight) {
            updatePending = true;
            return;
        }

        requestInFlight = true;
        updatePending = false;
        setLoading(true);
        setStatus("Updating structural preview…");

        try {
            const response = await fetch(previewUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "X-CSRF-Token": csrfToken
                },
                body: JSON.stringify({ choices: collectChoices() })
            });

            let payload = {};
            try {
                payload = await response.json();
            } catch (_error) {
                payload = {};
            }

            if (!response.ok) {
                throw new Error(payload.error || `Preview update failed (${response.status}).`);
            }

            if (previewImage && payload.preview_url) {
                const separator = payload.preview_url.includes("?") ? "&" : "?";
                previewImage.src = `${payload.preview_url}${separator}v=${Date.now()}`;
            }

            setStatus("Preview updated. Review the face before saving.", "success");
        } catch (error) {
            setStatus(
                error instanceof Error ? error.message : "Preview update failed.",
                "error"
            );
        } finally {
            requestInFlight = false;
            setLoading(false);

            if (updatePending) {
                sendPreviewUpdate();
            }
        }
    }

    function queuePreviewUpdate() {
        window.clearTimeout(debounceTimer);
        debounceTimer = window.setTimeout(sendPreviewUpdate, 160);
    }

    tabs.forEach((tab, index) => {
        tab.setAttribute("role", "tab");
        tab.setAttribute("aria-controls", tab.dataset.categoryTarget);
        tab.setAttribute("aria-selected", String(tab.classList.contains("active")));
        tab.setAttribute("tabindex", tab.classList.contains("active") ? "0" : "-1");

        tab.addEventListener("click", () => {
            openCategory(tab.dataset.categoryTarget);
        });

        tab.addEventListener("keydown", (event) => {
            if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) {
                return;
            }

            event.preventDefault();
            let nextIndex = index;
            if (event.key === "ArrowRight") nextIndex = (index + 1) % tabs.length;
            if (event.key === "ArrowLeft") nextIndex = (index - 1 + tabs.length) % tabs.length;
            if (event.key === "Home") nextIndex = 0;
            if (event.key === "End") nextIndex = tabs.length - 1;
            openCategory(tabs[nextIndex].dataset.categoryTarget, true);
        });
    });

    radios.forEach((radio) => {
        radio.addEventListener("change", () => {
            markSelectedCards();
            updateContradictionGuard();
            queuePreviewUpdate();
        });
    });

    form.addEventListener("submit", (event) => {
        const hasContradiction = updateContradictionGuard();
        if (
            hasContradiction
            && contradictionReason
            && contradictionReason.value.trim().length < 10
        ) {
            event.preventDefault();
            contradictionReason.setCustomValidity(
                "Enter at least 10 characters explaining the contradictory choice."
            );
            contradictionReason.reportValidity();
            contradictionReason.focus();
            setStatus("A reason is required before contradictory choices can be saved.", "error");
            return;
        }

        if (contradictionReason) {
            contradictionReason.setCustomValidity("");
        }
        window.clearTimeout(debounceTimer);
        if (finalizeButton) {
            finalizeButton.disabled = true;
            finalizeButton.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Saving Manual Face…';
        }
        setStatus("Saving the manually confirmed structural reference…");
    });

    if (contradictionReason) {
        contradictionReason.addEventListener("input", () => {
            contradictionReason.setCustomValidity("");
        });
    }

    markSelectedCards();
    updateContradictionGuard();
})();
