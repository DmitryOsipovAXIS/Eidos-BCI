/**
 * Gaze-driven focus and dwell-to-click navigation for EIDOS.
 */

import { GazeClient } from './gaze-client.js';
import {
    initApp,
    getDom,
    state,
    onScreenChange as subscribeScreenChange,
    enterMainMenu,
    powerOff,
    activateLeft,
    activateRight,
} from './app.js';

const FOCUS_CLASS = 'gaze-focused';
const DWELL_MS = 1000;

/** @type {'left' | 'right'} */
let focusedSide = 'left';

let centerSince = null;
let dwellTriggered = false;

let progressEl = null;
let progressTimer = null;

function clearFocus() {
    document.querySelectorAll(`.${FOCUS_CLASS}`).forEach((el) => el.classList.remove(FOCUS_CLASS));
}

function setFocus(element) {
    clearFocus();
    if (element) element.classList.add(FOCUS_CLASS);
}

function getFocusedElement() {
    const dom = getDom();
    if (state.currentScreen === 'splash') {
        return focusedSide === 'left' ? dom.splashEnter : dom.splashPower;
    }
    return focusedSide === 'left' ? dom.leftBtn : dom.rightBtn;
}

function applyFocusForSide(side) {
    focusedSide = side;
    setFocus(getFocusedElement());
}

function resetDwell() {
    centerSince = null;
    dwellTriggered = false;
    hideDwellProgress();
}

function showDwellProgress() {
    const el = getFocusedElement();
    if (!el || !centerSince) {
        hideDwellProgress();
        return;
    }
    if (!progressEl) {
        progressEl = document.createElement('div');
        progressEl.className = 'gaze-dwell-progress';
        document.querySelector('.app').appendChild(progressEl);
    }
    const rect = el.getBoundingClientRect();
    const appRect = document.querySelector('.app').getBoundingClientRect();
    progressEl.style.left = `${rect.left - appRect.left + rect.width / 2}px`;
    progressEl.style.top = `${rect.top - appRect.top + rect.height / 2}px`;
    progressEl.classList.add('visible');
}

function hideDwellProgress() {
    if (progressEl) progressEl.classList.remove('visible');
    clearInterval(progressTimer);
    progressTimer = null;
}

function updateDwellProgress() {
    if (!centerSince || dwellTriggered) return;
    const elapsed = Date.now() - centerSince;
    const ratio = Math.min(elapsed / DWELL_MS, 1);
    if (progressEl) {
        progressEl.style.setProperty('--dwell-scale', String(0.35 + ratio * 0.65));
        progressEl.style.opacity = String(0.35 + ratio * 0.65);
    }
    if (elapsed >= DWELL_MS && !dwellTriggered) {
        dwellTriggered = true;
        hideDwellProgress();
        activateFocused();
    }
}

function activateFocused() {
    if (state.currentScreen === 'splash') {
        if (focusedSide === 'left') enterMainMenu();
        else powerOff();
        return;
    }
    if (focusedSide === 'left') activateLeft();
    else activateRight();
}

function onGaze(gaze) {
    if (gaze === 'LEFT') {
        resetDwell();
        applyFocusForSide('left');
        return;
    }
    if (gaze === 'RIGHT') {
        resetDwell();
        applyFocusForSide('right');
        return;
    }
    if (gaze === 'CENTER') {
        if (!centerSince) {
            centerSince = Date.now();
            dwellTriggered = false;
            showDwellProgress();
            if (!progressTimer) {
                progressTimer = setInterval(updateDwellProgress, 50);
            }
        }
        updateDwellProgress();
        return;
    }
    resetDwell();
}

function setGazeStatus(text) {
    const dom = getDom();
    const base = dom.statusIndicator.dataset.baseLabel || dom.statusIndicator.innerText;
    if (!dom.statusIndicator.dataset.baseLabel) {
        dom.statusIndicator.dataset.baseLabel = base;
    }
    if (text === 'connected') {
        dom.statusIndicator.innerText = base;
        dom.statusIndicator.classList.remove('gaze-offline');
    } else {
        dom.statusIndicator.innerText = `GAZE · ${text.toUpperCase()}`;
        dom.statusIndicator.classList.add('gaze-offline');
    }
}

function syncBaseLabel() {
    const dom = getDom();
    const observer = new MutationObserver(() => {
        if (!getDom().statusIndicator.classList.contains('gaze-offline')) {
            dom.statusIndicator.dataset.baseLabel = dom.statusIndicator.innerText;
        }
    });
    observer.observe(dom.statusIndicator, { childList: true, characterData: true, subtree: true });
}

initApp();
subscribeScreenChange(() => {
    resetDwell();
    applyFocusForSide(focusedSide);
});
syncBaseLabel();
applyFocusForSide('left');

const client = new GazeClient(undefined, {
    onGaze,
    onStatus: setGazeStatus,
});
client.connect();
