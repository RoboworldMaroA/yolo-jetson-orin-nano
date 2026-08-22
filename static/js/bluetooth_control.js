/**
 * bluetooth_control.js
 * Wersja ze stałą sesją MediaSession dla iOS Safari
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
 * Tworzenie pętli audio aktywującej centrum sterowania na iOS
 */
// function createAudioElement() {
//   // Generowanie ciągłego szumu tła o bardzo niskiej głośności
//   const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
//   const buffer = audioCtx.createBuffer(1, audioCtx.sampleRate * 2, audioCtx.sampleRate);
//   const data = buffer.getChannelData(0);
//   for (let i = 0; i < buffer.length; i++) {
//     data[i] = (Math.random() * 2 - 1) * 0.001; // Znikomy szum niesłyszalny dla ucha
//   }

//   const source = audioCtx.createBufferSource();
//   source.buffer = buffer;
//   source.loop = true;

//   const destination = audioCtx.createMediaStreamDestination();
//   source.connect(destination);
//   source.start();

//   const audio = new Audio();
//   audio.srcObject = destination.stream;
//   audio.loop = true;
//   return audio;
// }
function createAudioElement() {
  // Generowanie absolutnej ciszy (zero szumu)
  const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  const buffer = audioCtx.createBuffer(1, audioCtx.sampleRate * 2, audioCtx.sampleRate);
  const data = buffer.getChannelData(0);
  
  // Wypełniamy bufor czystą ciszą
  for (let i = 0; i < buffer.length; i++) {
    data[i] = 0;
  }

  const source = audioCtx.createBufferSource();
  source.buffer = buffer;
  source.loop = true;

  const destination = audioCtx.createMediaStreamDestination();
  source.connect(destination);
  source.start();

  const audio = new Audio();
  audio.srcObject = destination.stream;
  audio.loop = true;
  return audio;
}


function initHondaRemote() {
  const btn = document.getElementById('startBtn');

  if (btn) {
    btn.innerText = '✅ Steering Wheel Controls Active';
    btn.style.background = '#2e7d32';
  }

  try {
    if (!dummyAudio) {
      dummyAudio = createAudioElement();
    }
    dummyAudio.play().catch(e => console.warn('[Bluetooth] Audio play warning:', e));
  } catch (e) {
    console.error('[Bluetooth] Audio context error:', e);
  }

  setupMediaSession();

  if ('speechSynthesis' in window) {
    speakModelName('Controls active');
  }
}

function setupMediaSession() {
  if ('mediaSession' in navigator) {
    navigator.mediaSession.metadata = new MediaMetadata({
      title: 'ADAS Controller',
      artist: 'Honda Remote System',
      album: 'Vision AI & Voice Control'
    });

    // Przechwytywanie komend
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

  const voiceText = MODEL_VOICE_NAMES[target.name] || `${target.name} active`;
  speakModelName(voiceText);

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