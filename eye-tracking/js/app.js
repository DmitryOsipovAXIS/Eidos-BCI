/**
 * EIDOS BCI prototype — screen state and UI actions (mouse + gaze).
 */

export const categories = [
    { id: 'help', icon: '🆘', title: 'HELP', desc: 'call assistance' },
    { id: 'movie', icon: '🎬', title: 'MOVIE', desc: 'watch films' },
    { id: 'music', icon: '🎵', title: 'MUSIC', desc: 'listen to music' },
];

export const state = {
    currentScreen: 'splash',
    currentCategoryIndex: 0,
};

const screenListeners = new Set();

/** Register a callback invoked whenever the active screen changes. */
export function onScreenChange(listener) {
    screenListeners.add(listener);
    return () => screenListeners.delete(listener);
}

function notifyScreenChange() {
    screenListeners.forEach((fn) => fn(state.currentScreen));
}

const dom = {};

export function initApp() {
    dom.splash = document.getElementById('splash');
    dom.mainInterface = document.getElementById('mainInterface');
    dom.splashEnter = document.getElementById('splashEnter');
    dom.splashPower = document.getElementById('splashPower');
    dom.exitButton = document.getElementById('exitButton');
    dom.leftBtn = document.getElementById('leftBtn');
    dom.rightBtn = document.getElementById('rightBtn');
    dom.leftIcon = document.getElementById('leftIcon');
    dom.leftLabel = document.getElementById('leftLabel');
    dom.leftHint = document.getElementById('leftHint');
    dom.rightIcon = document.getElementById('rightIcon');
    dom.rightLabel = document.getElementById('rightLabel');
    dom.rightHint = document.getElementById('rightHint');
    dom.categoryIcon = document.getElementById('categoryIcon');
    dom.categoryTitle = document.getElementById('categoryTitle');
    dom.categoryDesc = document.getElementById('categoryDesc');
    dom.instruction = document.getElementById('instruction');
    dom.statusIndicator = document.getElementById('statusIndicator');

    dom.splashEnter.addEventListener('click', enterMainMenu);
    dom.splashPower.addEventListener('click', powerOff);
    dom.exitButton.addEventListener('click', exitToPrevious);
    dom.leftBtn.addEventListener('click', activateLeft);
    dom.rightBtn.addEventListener('click', activateRight);

    dom.splash.classList.remove('hidden');
    dom.mainInterface.classList.remove('visible');
}

export function getDom() {
    return dom;
}

export function isCategoryScreen() {
    return state.currentScreen !== 'splash' && state.currentScreen !== 'mainMenu';
}

export function enterMainMenu() {
    if (state.currentScreen !== 'splash') return;
    state.currentScreen = 'mainMenu';
    dom.splash.classList.add('hidden');
    dom.mainInterface.classList.add('visible');
    updateMainMenuUI();
    notifyScreenChange();
}

export function powerOff() {
    if (state.currentScreen !== 'splash') return;
    alert('Eidos BCI · power off (simulated)');
}

export function exitToPrevious() {
    if (state.currentScreen === 'mainMenu') {
        state.currentScreen = 'splash';
        dom.splash.classList.remove('hidden');
        dom.mainInterface.classList.remove('visible');
        notifyScreenChange();
    } else if (isCategoryScreen()) {
        state.currentScreen = 'mainMenu';
        updateMainMenuUI();
        notifyScreenChange();
    }
}

export function activateLeft() {
    if (state.currentScreen === 'mainMenu') {
        state.currentCategoryIndex = (state.currentCategoryIndex + 1) % categories.length;
        updateMainMenuUI();
    } else if (isCategoryScreen()) {
        showCategoryFeedback('← back');
    }
}

export function activateRight() {
    if (state.currentScreen === 'mainMenu') {
        const category = categories[state.currentCategoryIndex];
        enterCategory(category.id);
    } else if (isCategoryScreen()) {
        showCategoryFeedback('✓ selected');
    }
}

export function updateMainMenuUI() {
    state.currentScreen = 'mainMenu';
    const cat = categories[state.currentCategoryIndex];

    dom.statusIndicator.innerText = 'MAIN MENU';
    dom.categoryIcon.innerText = cat.icon;
    dom.categoryTitle.innerText = cat.title;
    dom.categoryDesc.innerText = cat.desc;
    dom.instruction.innerText = '← next · select →';

    dom.leftLabel.innerText = 'NEXT';
    dom.leftIcon.innerText = '⟳';
    dom.leftHint.innerText = 'cycle';

    dom.rightLabel.innerText = 'SELECT';
    dom.rightIcon.innerText = '⏵';
    dom.rightHint.innerText = 'enter';
}

export function enterCategory(categoryId) {
    state.currentScreen = categoryId;
    dom.statusIndicator.innerText = categoryId.toUpperCase() + ' MODE';
    notifyScreenChange();

    if (categoryId === 'help') {
        dom.categoryIcon.innerText = '🆘';
        dom.categoryTitle.innerText = 'HELP';
        dom.categoryDesc.innerText = 'emergency call';
        dom.instruction.innerText = '← cancel · call →';
    } else if (categoryId === 'movie') {
        dom.categoryIcon.innerText = '🎬';
        dom.categoryTitle.innerText = 'MOVIE';
        dom.categoryDesc.innerText = 'browse films';
        dom.instruction.innerText = '← previous · select →';
    } else if (categoryId === 'music') {
        dom.categoryIcon.innerText = '🎵';
        dom.categoryTitle.innerText = 'MUSIC';
        dom.categoryDesc.innerText = 'choose playlist';
        dom.instruction.innerText = '← previous · play →';
    }

    dom.leftLabel.innerText = '←';
    dom.leftIcon.innerText = '←';
    dom.leftHint.innerText = 'back';

    dom.rightLabel.innerText = '✓';
    dom.rightIcon.innerText = '✓';
    dom.rightHint.innerText = categoryId === 'help' ? 'call' : categoryId === 'music' ? 'play' : 'select';
}

function showCategoryFeedback(action) {
    const originalDesc = dom.categoryDesc.innerText;
    dom.categoryDesc.innerText = action + ' ...';
    setTimeout(() => {
        if (state.currentScreen === 'help') dom.categoryDesc.innerText = 'emergency call';
        else if (state.currentScreen === 'movie') dom.categoryDesc.innerText = 'browse films';
        else if (state.currentScreen === 'music') dom.categoryDesc.innerText = 'choose playlist';
        else dom.categoryDesc.innerText = originalDesc;
    }, 500);
}
