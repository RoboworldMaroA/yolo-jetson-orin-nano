/**
 * bluetooth_control.js
 * Sterowanie aplikacją ADAS z kierownicy Hondy / słuchawek Bluetooth via Media Session API
 * Zoptymalizowane pod iOS (iPhone Safari) z pełną syntezą mowy (TTS)
 */

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

let currentModelIndex = 0;
let dummyAudio = null;
let lastVoiceTrigger = 0;

/**
 * Tworzenie cichego dźwięku w locie na iPhonie
 */
function createSilentAudio() {
  const audio = new Audio('data:audio/wav;base64,UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=');
  audio.loop = true;
  return audio;
}

/**
 * Wypowiadanie nazwy modelu (iOS kompatybilne)
 */
function speakModelName(text) {
  if ('speechSynthesis' in window) {
    window.speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.rate = 1.0;
    utterance.pitch = 1.0;

    window.speechSynthesis.speak(utterance);
  }
}

/**
 * Aktywowanie sterowania z kierownicy na iPhonie
 */
function initHondaRemote() {
  // 1. Odblokuj syntezę mowy w Safari na iPhonie
  if ('speechSynthesis' in window) {
    const unlockUtterance = new SpeechSynthesisUtterance('');
    window.speechSynthesis.speak(unlockUtterance);
  }

  // 2. Tworzenie i odtwarzanie audio w reakcji na kliknięcie na iPhonie
  if (!dummyAudio) {
    dummyAudio = createSilentAudio();
  }

  dummyAudio.play().then(() => {
    console.log('[Bluetooth] Remote control active on iOS!');

    const btn = document.getElementById('startBtn');
    if (btn) {
      btn.innerText = '✅ Steering Wheel Controls Active';
      btn.style.background = '#2e7d32';
    }

    // 3. Inicjalizacja Media Session API
    setupMediaSession();
  }).catch(err => {
    console.error('[Bluetooth] Audio unlock error on iOS:', err);
    alert('Błąd aktywacji audio na iOS. Kliknij przycisk ponownie.');
  });
}

function setupMediaSession() {
  if ('mediaSession' in navigator) {
    navigator.mediaSession.metadata = new MediaMetadata({
      title: 'ADAS Controller',
      artist: 'Honda Remote System',
      album: 'Vision AI & Voice Control'
    });

    navigator.mediaSession.setActionHandler('nexttrack', () => {
      switchModelBluetooth('next');
    });

    navigator.mediaSession.setActionHandler('previoustrack', () => {
      switchModelBluetooth('previous');
    });

    navigator.mediaSession.setActionHandler('play', () => {
      triggerVoiceControl();
    });

    navigator.mediaSession.setActionHandler('pause', () => {
      triggerVoiceControl();
    });
  } else {
    alert('Media Session API nie jest wspierane w tej przeglądarce.');
  }
}

function switchModelBluetooth(direction) {
  if (direction === 'next') {
    currentModelIndex = (currentModelIndex + 1) % MODEL_MAPPING.length;
  } else if (direction === 'previous') {
    currentModelIndex = (currentModelIndex - 1 + MODEL_MAPPING.length) % MODEL_MAPPING.length;
  }

  const target = MODEL_MAPPING[currentModelIndex];
  const buttonElem = document.getElementById(target.id);

  console.log(`[Bluetooth] Switching model (${currentModelIndex + 1}/${MODEL_MAPPING.length}): ${target.name}`);

  // Synteza mowy
  const voiceText = MODEL_VOICE_NAMES[target.name] || `${target.name} active`;
  speakModelName(voiceText);

  // Zmiana modelu w HTML / backendzie
  if (buttonElem && !buttonElem.disabled) {
    buttonElem.click();
  } else if (typeof window.switchModel === 'function') {
    window.switchModel(target.name);
  }
}

function triggerVoiceControl() {
  const now = Date.now();
  if (now - lastVoiceTrigger < 800) return;
  lastVoiceTrigger = now;

  console.log('[Bluetooth] Middle button pressed – triggering Whisper microphone!');
  const voiceBtn = document.getElementById('record-btn-voice');

  if (voiceBtn) {
    voiceBtn.click();
  } else {
    console.warn('[Bluetooth] Voice record button (#record-btn-voice) not found!');
  }
}