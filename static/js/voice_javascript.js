import { pipeline, env } from './transformers.min.js';

// Konfiguracja pracy OFFLINE z dysku Jetsona
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = '/static/models/';

let transcriber = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

// 1. Ładowanie modelu Whisper (skonfigurowanego na język angielski)
async function initWhisper() {
    console.log("⏳ Loading Whisper model...");
    const statusEl = document.getElementById('status');
    
    try {
        transcriber = await pipeline('automatic-speech-recognition', 'whisper-tiny', {
            quantized: true
        });
        console.log("✅ Whisper model loaded and ready!");
        if (statusEl) statusEl.innerText = "✅ Whisper model ready!";
    } catch (err) {
        console.error("❌ Error loading Whisper model:", err);
        if (statusEl) statusEl.innerText = "❌ Model loading error.";
    }
}

// 2. Główna logika sterowania komendami głosowymi
document.addEventListener('DOMContentLoaded', () => {
    initWhisper();

    const recordBtn = document.getElementById('record-btn');

    // Inicjalizacja strumienia mikrofonu z MacBooka
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                console.log("🎤 Microphone access granted.");
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = e => {
                    if (e.data && e.data.size > 0) {
                        audioChunks.push(e.data);
                    }
                };

                mediaRecorder.onstop = async () => {
                    console.log("⏹️ Recording stopped. Processing audio...");
                    const statusEl = document.getElementById('status');
                    if (statusEl) statusEl.innerText = "🧠 Processing speech...";

                    if (audioChunks.length === 0) {
                        console.warn("⚠️ No audio recorded.");
                        if (statusEl) statusEl.innerText = "⚠️ No audio detected.";
                        return;
                    }

                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    audioChunks = [];

                    if (!transcriber) {
                        alert("Whisper model is not ready yet!");
                        return;
                    }

                    try {
                        const audioUrl = URL.createObjectURL(audioBlob);
                        
                        // Transkrypcja w języku angielskim
                        const output = await transcriber(audioUrl, { 
                            language: 'english',
                            task: 'transcribe'
                        });
                        
                        const text = output.text.toLowerCase().trim();
                        console.log("🗣️ Recognized command:", text);

                        const outputEl = document.getElementById('output');
                        if (outputEl) outputEl.innerText = "Recognized: " + text;

                        // Parsowanie i wywoływanie akcji w aplikacji
                        processVoiceCommand(text);

                    } catch (err) {
                        console.error("❌ Transcription error:", err);
                        if (statusEl) statusEl.innerText = "❌ Speech processing error.";
                    }
                };

                // Listener dla przycisku nagrywania
                if (recordBtn) {
                    recordBtn.addEventListener('click', (e) => {
                        e.preventDefault();
                        if (!isRecording) {
                            audioChunks = [];
                            mediaRecorder.start();
                            isRecording = true;
                            recordBtn.innerText = "🛑 Stop recording";
                            recordBtn.style.backgroundColor = "#b00020";
                            const statusEl = document.getElementById('status');
                            if (statusEl) statusEl.innerText = "🎙️ Listening... Speak command";
                        } else {
                            mediaRecorder.stop();
                            isRecording = false;
                            recordBtn.innerText = "🎙️ Start recording";
                            recordBtn.style.backgroundColor = "";
                        }
                    });
                }
            })
            .catch(err => {
                console.error("❌ Microphone access error:", err);
            });
    }
});

// 3. Dopasowywanie tekstu angielskiego do przycisków / funkcji w kodzie HTML
function processVoiceCommand(cmd) {
    const statusEl = document.getElementById('status');

    // Model switching (przekazywanie exact ID modeli z Twojego JS)
    if (cmd.includes("road detection segmentation") || cmd.includes("road segmentation")) {
        triggerButtonClick('btn-road-detection-segmentation');
    } else if (cmd.includes("road detection") || cmd.includes("road")) {
        triggerButtonClick('btn-road-detection');
    } else if (cmd.includes("improved irish plate") || cmd.includes("improved plate")) {
        triggerButtonClick('btn-improved-irish-plate-recognition');
    } else if (cmd.includes("irish plate") || cmd.includes("irish")) {
        triggerButtonClick('btn-irish-plate-recognition');
    } else if (cmd.includes("plate recognition") || cmd.includes("plate") || cmd.includes("license plate")) {
        triggerButtonClick('btn-plate-recognition');
    } else if (cmd.includes("lprnet") || cmd.includes("anpr")) {
        triggerButtonClick('btn-lprnet-anpr');
    } else if (cmd.includes("custom model") || cmd.includes("yolo 11")) {
        triggerButtonClick('btn-custom-model');
    } else if (cmd.includes("pose estimation") || cmd.includes("pose")) {
        triggerButtonClick('btn-pose');
    } else if (cmd.includes("segmentation")) {
        triggerButtonClick('btn-segmentation');
    } else if (cmd.includes("object detection") || cmd.includes("detection") || cmd.includes("objects")) {
        triggerButtonClick('btn-detect');
    } 
    
    // Utwalanie nagrania / robienie zdjęć
    else if (cmd.includes("start recording") || cmd.includes("record video") || cmd.includes("stop recording")) {
        triggerButtonClick('record-toggle');
    } else if (cmd.includes("save image") || cmd.includes("take photo") || cmd.includes("snapshot")) {
        triggerButtonClick('save-image');
    } 
    
    // Pozycja kamery
    else if (cmd.includes("car mode") || cmd.includes("flip camera") || cmd.includes("upside down")) {
        triggerButtonClick('camera-car');
    } else if (cmd.includes("normal mode") || cmd.includes("normal camera")) {
        triggerButtonClick('camera-normal');
    } 
    
    else {
        console.warn("⚠️ Command not recognized:", cmd);
        if (statusEl) statusEl.innerText = `⚠️ Unrecognized command: "${cmd}"`;
    }
}

// Helper wywołujący zdarzenie `.click()` na przycisku
function triggerButtonClick(buttonId) {
    const btn = document.getElementById(buttonId);
    if (btn) {
        console.log(`🚀 Executing click on #${buttonId}`);
        btn.click();
    } else {
        console.error(`❌ Button with ID #${buttonId} not found in DOM.`);
    }
}