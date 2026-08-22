/**
 * bluetooth_control.js
 * Sterowanie aplikacją ADAS z kierownicy Hondy / słuchawek Bluetooth via Media Session API
 */


// Słownik mapujący ID przycisków na nazwy modeli przyjmowane przez funkcję switchModel()
const MODEL_MAPPING = [
  { id: 'btn-detect', name: 'detection' },
  { id: 'btn-pose', name: 'pose' },
  { id: 'btn-segmentation', name: 'segmentation' },
  { id: 'btn-custom-model', name: 'custom-model' },
  { id: 'btn-plate-recognition', name: 'plate-recognition' },
  { id: 'btn-irish-plate-recognition', name: 'irish-plate-recognition' },
  { id: 'btn-improved-irish-plate-recognition', name: 'improved-irish-plate-recognition' },
  { id: 'btn-lprnet-anpr', name: 'lprnet-anpr' },
  { id: 'btn-road-detection', name: 'road-detection' },
  { id: 'btn-road-detection-segmentation', name: 'road-detection-segmentation' }
];

let currentModelIndex = 0; // Start od 'detection'

const dummyAudio = new Audio('data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=');
dummyAudio.loop = true;

/**
 * Inicjalizacja sterowania Bluetooth – wywoływana po kliknięciu przycisku na stronie
 */
function initHondaRemote() {
  dummyAudio.play().then(() => {
    console.log('[Bluetooth] Steering wheel control enabled!');

    const btn = document.getElementById('startBtn');
    if (btn) {
      btn.innerText = '✅ Steering Wheel Controls Active';
      btn.style.background = '#2e7d32';
    }
  }).catch(err => console.error('[Bluetooth] Audio playback error:', err));

  if ('mediaSession' in navigator) {
    navigator.mediaSession.metadata = new MediaMetadata({
      title: 'ADAS Controller',
      artist: 'Honda Remote System',
      album: 'Vision AI & Voice Control'
    });

    // Strzałka w prawo -> Następny model
    navigator.mediaSession.setActionHandler('nexttrack', () => {
      switchModelBluetooth('next');
    });

    // Strzałka w lewo -> Poprzedni model
    navigator.mediaSession.setActionHandler('previoustrack', () => {
      switchModelBluetooth('previous');
    });

    // Środkowy przycisk -> Mikrofon
    navigator.mediaSession.setActionHandler('play', () => {
      triggerVoiceControl();
    });

    navigator.mediaSession.setActionHandler('pause', () => {
      triggerVoiceControl();
    });
  } else {
    console.warn('[Bluetooth] Media Session API is not supported in this browser.');
  }
}

/**
 * Przełącza modele AI w pętli i wywołuje akcję w aplikacji
 * @param {string} direction - 'next' lub 'previous'
 */
// function switchModelBluetooth(direction) {
//   if (direction === 'next') {
//     currentModelIndex = (currentModelIndex + 1) % MODEL_MAPPING.length;
//   } else if (direction === 'previous') {
//     currentModelIndex = (currentModelIndex - 1 + MODEL_MAPPING.length) % MODEL_MAPPING.length;
//   }

//   const target = MODEL_MAPPING[currentModelIndex];
//   const buttonElem = document.getElementById(target.id);

//   console.log(`[Bluetooth] Switching model (${currentModelIndex + 1}/${MODEL_MAPPING.length}): ${target.name}`);

//   if (buttonElem && !buttonElem.disabled) {
//     // Jeśli przycisk nie jest zablokowany, wywołujemy kliknięcie
//     buttonElem.click();
//   } else if (typeof window.switchModel === 'function') {
//     // Jeśli przycisk jest zablokowany (disabled), bezpośrednio wywołujemy funkcję switchModel(name) z HTML
//     window.switchModel(target.name);
//   } else {
//     console.warn(`[Bluetooth] Could not switch to model: ${target.name}`);
//   }
// }

function switchModelBluetooth(direction) {
  if (direction === 'next') {
    currentModelIndex = (currentModelIndex + 1) % MODEL_MAPPING.length;
  } else if (direction === 'previous') {
    currentModelIndex = (currentModelIndex - 1 + MODEL_MAPPING.length) % MODEL_MAPPING.length;
  }

  const target = MODEL_MAPPING[currentModelIndex];
  const buttonElem = document.getElementById(target.id);

  console.log(`[Bluetooth] Switching model (${currentModelIndex + 1}/${MODEL_MAPPING.length}): ${target.name}`);

  // 1. Wypowiedz informację głosową
  const voiceText = MODEL_VOICE_NAMES[target.name] || `${target.name} active`;
  speakModelName(voiceText);

  // 2. Aktywuj model na backendzie
  if (buttonElem && !buttonElem.disabled) {
    buttonElem.click();
  } else if (typeof window.switchModel === 'function') {
    window.switchModel(target.name);
  } else {
    console.warn(`[Bluetooth] Could not switch to model: ${target.name}`);
  }
}

/**
 * Aktywacja rozpoznawania mowy
 */
// function triggerVoiceControl() {
//   console.log('[Bluetooth] Middle button pressed – triggering microphone!');

//   // Szukamy przycisku nagrywania w Twoim kodzie HTML
//   const recordBtn = document.getElementById('recordBtn') || document.getElementById('record-toggle');

//   if (recordBtn) {
//     recordBtn.click();
//   } else {
//     console.warn('[Bluetooth] Record button not found!');
//   }
// }

let lastVoiceTrigger = 0;

/**
 * Aktywacja / zatrzymanie nagrywania głosu (Whisper)
 */
function triggerVoiceControl() {
  const now = Date.now();
  // Zabezpieczenie przed podwójnym kliknięciem przy przełączeniu play/pause (min 800ms przerwy)
  if (now - lastVoiceTrigger < 800) {
    return;
  }
  lastVoiceTrigger = now;

  console.log('[Bluetooth] Middle button pressed – triggering Whisper microphone!');

  // Dedykowany przycisk mikrofonu z pliku voice detection
  const voiceBtn = document.getElementById('record-btn-voice');

  if (voiceBtn) {
    // Symulacja kliknięcia na przycisk mikrofonu
    voiceBtn.click();
  } else {
    console.warn('[Bluetooth] Could not find #record-btn-voice element!');
  }
}





/**
 * Funkcja wypowiadająca na głos tekst po angielsku
 * @param {string} text - Tekst do przeczytania przez lektora
 */
function speakModelName(text) {
  if ('speechSynthesis' in window) {
    // Anulujemy poprzednie wypowiedzi, jeśli strzałka była klikana szybko kilka razy
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US'; // Język angielski
    utterance.rate = 1.0;     // Prędkość mowy (1.0 = normalna)
    utterance.pitch = 1.0;    // Ton głosu

    window.speechSynthesis.speak(utterance);
  }
}

// Przyjazne dla ucha nazwy modeli do wypowiedzenia na głos
const MODEL_VOICE_NAMES = {
  'detection': 'Object Detection active',
  'pose': 'Pose Estimation active',
  'segmentation': 'Segmentation active',
  'custom-model': 'Custom Model active',
  'plate-recognition': 'Plate Recognition active',
  'irish-plate-recognition': 'Irish Plate active',
  'improved-irish-plate-recognition': 'Improved Irish Plate active',
  'lprnet-anpr': 'LPR Net active',
  'road-detection': 'Road Detection active',
  'road-detection-segmentation': 'Road Detection Segmentation active'
};