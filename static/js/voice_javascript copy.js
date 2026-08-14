import { pipeline, env } from './transformers.min.js';

// Konfiguracja pracy OFFLINE
env.allowRemoteModels = false;
env.allowLocalModels = true;
env.localModelPath = '/static/models/';

let transcriber = null;
let mediaRecorder = null;
let audioChunks = [];
let isRecording = false;

async function initModel() {
    console.log("⏳ Inicjalizacja modelu Whisper...");
    const statusEl = document.getElementById('status');
    
    try {
        transcriber = await pipeline('automatic-speech-recognition', 'whisper-tiny', {
            quantized: true
        });
        console.log("✅ Model Whisper został pomyślnie załadowany i jest gotowy!");
        if (statusEl) statusEl.innerText = "✅ Model Whisper gotowy!";
    } catch (err) {
        console.error("❌ Błąd podczas ładowania modelu Whisper:", err);
        if (statusEl) statusEl.innerText = "❌ Błąd ładowania modelu.";
    }
}

document.addEventListener('DOMContentLoaded', () => {
    initModel();

    const recordBtn = document.getElementById('record-btn');

    // Inicjalizacja mikrofonu
    if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(stream => {
                console.log("🎤 Dostęp do mikrofonu przyznany.");
                mediaRecorder = new MediaRecorder(stream);

                mediaRecorder.ondataavailable = e => {
                    if (e.data && e.data.size > 0) {
                        audioChunks.push(e.data);
                    }
                };

                mediaRecorder.onstop = async () => {
                    console.log("⏹️ Zatrzymano nagrywanie. Rozpoczynam przetwarzanie...");
                    const statusEl = document.getElementById('status');
                    if (statusEl) statusEl.innerText = "🧠 Przetwarzanie mowy...";

                    if (audioChunks.length === 0) {
                        console.warn("⚠️ Brak nagranego dźwięku.");
                        if (statusEl) statusEl.innerText = "⚠️ Brak dźwięku. Spróbuj ponownie.";
                        return;
                    }

                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    audioChunks = [];

                    if (!transcriber) {
                        alert("Model Whisper nie jest jeszcze gotowy!");
                        return;
                    }

                    try {
                        const audioUrl = URL.createObjectURL(audioBlob);
                        const output = await transcriber(audioUrl, { language: 'polish' });
                        
                        const text = output.text.toLowerCase().trim();
                        console.log("🗣️ Rozpoznana komenda:", text);

                        const outputEl = document.getElementById('output');
                        if (outputEl) outputEl.innerText = "Rozpoznano: " + text;
                        if (statusEl) statusEl.innerText = "✅ Gotowe.";

                        // Dopasowanie komend
                        if (text.includes("segmentacja") || text.includes("segmentację")) {
                            sendModelChange('segmentation');
                        } else if (text.includes("detekcja") || text.includes("obiektów")) {
                            sendModelChange('object_detection');
                        } else if (text.includes("tablice") || text.includes("rejestracja")) {
                            sendModelChange('plate_recognition');
                        }
                    } catch (err) {
                        console.error("❌ Błąd transkrypcji:", err);
                        if (statusEl) statusEl.innerText = "❌ Błąd przetwarzania mowy.";
                    }
                };

                // Obsługa przycisku
                if (recordBtn) {
                    recordBtn.addEventListener('click', (e) => {
                        e.preventDefault();
                        if (!isRecording) {
                            // Start nagrywania
                            audioChunks = [];
                            mediaRecorder.start();
                            isRecording = true;
                            recordBtn.innerText = "🛑 Stop recording";
                            const statusEl = document.getElementById('status');
                            if (statusEl) statusEl.innerText = "🎙️ Nagrywanie... Mów komendę";
                        } else {
                            // Stop nagrywania
                            mediaRecorder.stop();
                            isRecording = false;
                            recordBtn.innerText = "🎙️ Start recording";
                        }
                    });
                }
            })
            .catch(err => {
                console.error("❌ Błąd mikrofonu:", err);
            });
    }
});

function sendModelChange(modelType) {
    console.log("🚀 Wysyłanie żądania zmiany modelu na:", modelType);
    fetch('/set_model', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model: modelType })
    });
}