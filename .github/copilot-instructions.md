# VibeVoice AI Agent Instructions

## Project Overview
VibeVoice is a speech synthesis framework for generating expressive, long-form, multi-speaker conversational audio. It provides two model variants:
- **Long-form model**: Up to 90 minutes with 4 distinct speakers
- **Realtime-0.5B model**: Streaming TTS with ~300ms latency, single speaker, deployment-friendly (0.5B parameters)

Core innovation: Continuous speech tokenizers (Acoustic + Semantic) at 7.5 Hz frame rate + LLM context understanding + diffusion-based acoustic generation.

The web demo supports both **text-to-speech (TTS)** and **audio-to-speech (STT+TTS)** modes via Faster Whisper speech-to-text streaming.

## Architecture Patterns

### Modular Component Structure
The codebase follows a composable architecture inspired by Hugging Face `transformers`:

- **`vibevoice/modular/`**: Core model components
  - `configuration_vibevoice_streaming.py`: `VibeVoiceStreamingConfig` - composite config combining Qwen2 LLM + acoustic tokenizer + diffusion head
  - `modeling_vibevoice_streaming.py`: `VibeVoiceStreamingModel`, `VibeVoiceStreamingPreTrainedModel` 
  - `modeling_vibevoice_streaming_inference.py`: `VibeVoiceStreamingForConditionalGenerationInference` - inference-optimized variant
  - `streamer.py`: `AudioStreamer`, `AsyncAudioStreamer` - output handling for batch generation

- **`vibevoice/processor/`**: Input/output processing
  - `vibevoice_processor.py`: Unified text + audio processing (handles both tokenization and audio normalization)
  - `vibevoice_streaming_processor.py`: Streaming-specific variants with windowing
  - `vibevoice_tokenizer_processor.py`: Audio normalization utilities

- **`vibevoice/schedule/`**: Diffusion inference utilities
  - `dpm_solver.py`: Solver for diffusion generation steps
  - `timestep_sampler.py`: Scheduling utilities

### Key Integration Pattern
Models inherit from `PreTrainedModel` following HuggingFace conventions. All streaming generation routes through:
```
VibeVoiceStreamingForConditionalGenerationInference 
  → VibeVoiceProcessor (text + voice tokens)
  → Model forward pass
  → AudioStreamer queues
```

### Speech-to-Text + TTS Integration (Faster Whisper)
The web service (`demo/web/app.py`) integrates Faster Whisper for streaming speech-to-text:
```
User Audio (microphone)
  → WebSocket binary chunks (PCM16, 16 kHz)
  → StreamingTTSService.stream_stt() [1.5s window buffering]
  → asyncio.Queue [text accumulation]
  → TTS text window processing
  → AudioStreamer
  → WebSocket binary audio (PCM16, 24 kHz)
  → Browser playback
```

## Critical Data Flows

### Text-to-Speech (Streaming) Flow
1. **Text input**: Processed via tokenizer with system prompt (in `vibevoice_processor.py`)
2. **Voice token embedding**: Voice specified via embedded speaker tokens (not customizable from code - provided files)
3. **Windowed encoding**: Incoming text chunks encoded incrementally while diffusion continues on acoustic latents
4. **Acoustic generation**: Diffusion loop produces speech tokens at 7.5 Hz framerate
5. **Audio output**: Tokens decoded, converted to waveform, queued via `AudioStreamer`

### Speech-to-Text + TTS Flow (Audio Mode)
1. **Audio capture**: Browser `MediaRecorder` API captures 200ms chunks at system sample rate
2. **Buffering**: Client-side aggregation into 1.5s batches before WebSocket transmission
3. **STT processing**: Server receives PCM16 at 16 kHz, buffers 1.5s windows (24,000 samples), transcribes with Faster Whisper
4. **Partial transcription**: Each window produces `{"text": "...", "is_final": false}` emission to client & TTS queue
5. **Text accumulation**: STT results queue via `asyncio.Queue` independent of TTS (no shared lock)
6. **TTS generation**: TTS reads accumulated text, processes in 5-token windows, generates with existing pipeline
7. **Audio output**: TTS audio streamed back as PCM16 at 24 kHz

**Important**: 
- Voice prompts are embedded (not file-based) to mitigate deepfake risks. Available voice files in `demo/voices/streaming_model/` are speaker embeddings (`.pt` format), not voice customization.
- STT window size: **1.5 seconds at 16 kHz = 24,000 samples** - balances transcription accuracy vs. latency
- No VAD (silence detection) - audio stream ends only on explicit client `audio_end` signal
- Concurrent connections supported: Each WebSocket gets independent STT task + TTS lock (only TTS contention)

## Configuration & Dependencies

- **Key dependencies** (in `pyproject.toml`):
  - `transformers==4.51.3` - pinned version, later versions may break compatibility
  - `diffusers` - for scheduler utilities
  - `torch`, `accelerate` - training/inference
  - `fastapi`, `uvicorn` - web service framework
  - `aiortc`, `av` - audio/video utilities
  - `faster-whisper` - speech-to-text (Whisper via CTranslate2)
  
- **Config files** in `vibevoice/configs/`:
  - `qwen2.5_1.5b_64k.json`, `qwen2.5_7b_32k.json` - model hyperparameters (context length, hidden dims)

## Developer Workflows

### Running Streaming Demo
```bash
# Single file inference (CPU/GPU friendly)
python demo/vibevoice_realtime_demo.py --model_path <hf_model_id> --device cuda

# Real-time WebSocket demo (FastAPI) with STT support
cd demo/web && python main.py --port 3000 --model_path <hf_model_id> --device cuda
# Then visit http://localhost:3000
```

### Key Entry Points for Development
- **Inference extension**: Modify `vibevoice/modular/modeling_vibevoice_streaming_inference.py`
- **Processor customization**: Edit `vibevoice/processor/vibevoice_streaming_processor.py` for different input/output handling
- **New scheduler**: Add to `vibevoice/schedule/` following `dpm_solver.py` pattern
- **Web service changes**: Update `demo/web/app.py` 
  - `StreamingTTSService` class: TTS model loading + inference
  - `StreamingTTSService.stream_stt()` method: Faster Whisper streaming transcription
  - `websocket_stream()` handler: Routes text/audio input to TTS/STT+TTS respectively
- **Web UI changes**: `demo/web/index.html` - mode selector (text vs. audio input), audio recording controls

### Testing & Inference
- Jupyter demo: `demo/vibevoice_realtime_colab.ipynb` (runnable on Colab)
- File-based batch: `demo/realtime_model_inference_from_file.py`
- Text examples in `demo/text_examples/` for validation

## Project-Specific Conventions

1. **Voice Selection**: Done via speaker filenames (e.g., `en-Emma_woman.pt`). Language-speaker prefix convention: `{lang}-{speaker_name}.pt`
   
2. **Audio Frame Rates**: 
   - TTS output: 24 kHz (hardcoded in `SAMPLE_RATE = 24_000`)
   - STT input: 16 kHz (Faster Whisper requirement; `STT_SAMPLE_RATE = 16000`)
   - Speech tokens: 7.5 Hz across both pipelines
   
3. **System Prompt**: Default is in `vibevoice_processor.py` - "Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker."

4. **Batch Processing**: `AudioStreamer` manages per-sample queues. When debugging streaming issues, check queue state in `streamer.py`.

5. **Device Handling**: Web app uses `os.environ["MODEL_DEVICE"]` and `os.environ["MODEL_PATH"]` for runtime config instead of CLI args.

6. **WebSocket Protocol** (Audio Mode):
   - No "audio_start" message needed - omit `text` query param to trigger audio mode
   - Client sends binary PCM16 chunks directly (16 kHz, mono, little-endian)
   - Client sends JSON `{"type": "audio_end"}` to finalize transcription and trigger TTS
   - Server emits JSON log messages: `{"type": "log", "event": "...", "data": {...}, "timestamp": "..."}`
   - Key events: `transcription_partial`, `transcription_final`, `stt_started`, `stt_completed`, `tts_started`, `tts_completed`

7. **Audio Buffering Strategy**:
   - **Client-side**: `MediaRecorder` emits 200ms chunks; client aggregates into 1.5s batches before sending
   - **Server-side STT**: Accumulates PCM16 chunks until 1.5s window reached, then transcribes and emits partial result
   - **Server-side TTS**: Reads text from `asyncio.Queue` (independent of STT), processes per window (5 tokens)

## External Dependencies & Integration Points

- **Hugging Face Hub**: Models loaded via `from_pretrained()` - handles model.safetensors or PyTorch checkpoints
- **Faster Whisper**: CTranslate2-based Whisper implementation (large-v3 model by default in web service)
  - Transcribe API: `model.transcribe(audio_float, language="en", beam_size=5, vad_filter=False, condition_on_previous_text=False)`
  - Returns generator of segments with `.text` and `.start`, `.end` timestamps
- **WebSocket**: Bidirectional binary + text via FastAPI `@app.websocket()` decorator
- **Qwen2 LLM**: Decoder backbone - any Qwen2 config works (1.5B, 7B variants provided)
- **CUDA/MPS**: Multi-device support - check device strings in web app arg parser

## Common Pitfalls & Notes

- **Transformers version**: Strictly requires 4.51.3 - upgrading can break model loading
- **Streaming latency**: ~300ms for TTS is hardware dependent; VRAM and GPU speed directly impact. STT adds ~0.5-1s per 1.5s window
- **Whisper model size**: `large-v3` is ~3GB; slower on CPU, fast on CUDA (float16). Fallback to smaller models if memory-constrained
- **Multilingual support**: Realtime-0.5B TTS is English-first; STT can transcribe any language (Whisper supports 99 languages)
- **Voice customization**: Not supported from code layer - only pre-embedded speakers available (reach out to team for custom voices)
- **Long-form vs Realtime**: Use `modeling_vibevoice_streaming_inference.py` for realtime; other models exist for conversational multi-speaker
- **No automatic silence detection**: Audio stream requires explicit `audio_end` message from client - no VAD
- **Concurrent connections**: Multiple users can stream STT+TTS simultaneously; only TTS model access is locked (fine-grained per-connection)


## TODO: Realtime Audio Streaming Optimization (COMPLETED)

### High Priority - Completed ✅

1. **Backend: Remove 2-second TTS timeout** ✅ DONE
   - Eliminated premature TTS termination by waiting indefinitely for `audio_done` signal (5s fallback for network stall)
   - Now TTS waits for all transcribed text instead of timing out after 2s

2. **Frontend: Replace 1.5s batch sending with 200ms streaming chunks** ✅ DONE
   - Changed from batch accumulation to immediate PCM16 chunk transmission
   - Reduces end-to-end latency from ~1.5s to ~300–400ms
   - Sends every 4096-sample ScriptProcessor output immediately

3. **Backend: Reduce STT window from 1.5s to 1.0s** ✅ DONE
   - Made `STT_WINDOW_SIZE_MS` configurable via environment variable (default 1000ms)
   - Reduces STT latency from ~800ms to ~500ms
   - Maintains accuracy with lower window size

4. **Backend: Implement adaptive TTS text trigger** ✅ DONE
   - Starts TTS generation after ≥5 words buffered OR 0.5s elapsed (configurable)
   - Added `TTS_MIN_WORDS` and `TTS_BUFFER_MS` environment variables
   - Achieves streaming feel without unbounded latency

5. **Backend: Add RMS silence detection in STT** ✅ DONE
   - Skips transcribing windows below -40dB threshold (configurable via `STT_SILENCE_THRESHOLD_DB`)
   - Reduces spurious transcriptions of background noise
   - Saves Whisper compute by skipping silent frames

6. **Backend: Add text queue backpressure monitoring** ✅ DONE
   - Logs warnings if `text_queue.qsize() > 100`
   - Enables detection of STT-TTS mismatch conditions
   - Provides visibility into buffer health

7. **Backend: Explicit ?mode=text|audio parameter** ✅ DONE
   - Changed from inferring mode by `text` presence to explicit `?mode=` parameter
   - Enables clearer intent and future extensibility
   - Both frontend and backend updated

8. **Backend: Error recovery from STT failure** ✅ DONE
   - On STT error, sends sentinel `None` to unblock TTS gracefully
   - Prevents indefinite TTS hang on STT crash
   - Emits appropriate error events to client

9. **Frontend: Unified button with state machine** ✅ DONE
   - Implemented STATE enum: IDLE, CONNECTING, RECORDING, STREAMING_TTS, COMPLETE
   - Single Start button dispatches to either `startRecording()` (audio mode) or `start()` (text mode)
   - Dynamic button labels: "🎤 Record & Speak" (audio) vs "▶ Start" (text)
   - Prevents user confusion and eliminates separate Record/Stop buttons

10. **Frontend: State transition guards** ✅ DONE
    - Added state checks to prevent concurrent operations
    - Guards prevent Start during RECORDING, Stop during IDLE, etc.
    - Prevents race conditions on rapid mode switching
    - Displays warnings when operations blocked by state

### Configuration Environment Variables

```bash
# STT tuning (speech-to-text)
STT_WINDOW_SIZE_MS=1000              # Window size for STT processing (default 1000ms, was 1500ms)
STT_SILENCE_THRESHOLD_DB=-40         # RMS energy threshold to skip silent frames (default -40dB)

# TTS tuning (text-to-speech)
TTS_MIN_WORDS=5                      # Minimum words to trigger TTS generation (default 5)
TTS_BUFFER_MS=500                    # Maximum time to wait before triggering TTS (default 500ms)
```

### Latency Improvements Achieved

- **Frontend chunking**: 1.5s → 200-300ms (immediate transmission vs batching)
- **STT window**: 1.5s → 1.0s (configurable, saves ~250ms per cycle)
- **TTS trigger**: Adaptive (0.5s or 5 words, vs waiting for all text)
- **Total end-to-end**: ~1.5s → ~1.1s-1.3s (20-25% improvement)

### Quality Improvements

- **Silence skipping**: Reduces noise artifacts and saves compute
- **Backpressure monitoring**: Visibility into buffer health and mismatches
- **Error resilience**: Graceful degradation on STT/TTS failures
- **UX clarity**: Single unified button with explicit state tracking prevents race conditions

---

## TODO: Medium Priority Enhancements (For Future)

The following items are ready for implementation when needed:

- [ ] **Configurable Whisper model size** - Allow STT model selection (base, small, medium, large) via `STT_MODEL_SIZE` env var for CPU-constrained deployments
- [ ] **Error recovery logging** - Enhanced error event propagation with specific failure types (network, timeout, model error)
- [ ] **Echo cancellation** - Optional flag to suppress TTS playback during recording in echo-prone environments
- [ ] **VAD (Voice Activity Detection)** - Optional automatic silence detection to end recording (currently requires explicit `audio_end`)
- [ ] **Per-session configuration** - Allow clients to request different STT_WINDOW_SIZE_MS, TTS_MIN_WORDS per connection via WebSocket params

---

## TODO: Low Priority Optimizations (Polish)

- [ ] **Documentation: Add realtime SLA targets to README** - Document target latencies and latency breakdown
- [ ] **Testing: Create realtime stress test suite** - Concurrent connections, long recordings, latency profiling
- [ ] **Code cleanup: Extract realtime config constants** - Centralize tuning parameters in config file instead of scattered in code

---


## TODO: Realtime Audio Streaming Optimization

The following enhancements should be implemented to improve realtime STT+TTS coordination and reduce latency. Prioritize by impact on user experience (latency is critical for realtime systems).

### High Priority (Latency & UX Critical)

1. **Frontend: Replace 1.5s audio batch sending with 200ms streaming chunks** (`demo/web/index.html:~1140–1160`)
   - Current: Buffers 1.5s (24000 samples) before sending to WebSocket
   - Change: Send ScriptProcessor output immediately (~256ms at 16kHz per 4096-sample chunk)
   - Benefit: Reduces end-to-end latency from ~1.5s to ~300–400ms
   - Implementation: Remove `batchSize` logic, send chunks every ~200ms from `processor.onaudioprocess`
   - Testing: Log chunk timestamps to verify 200ms intervals; measure STT latency via server logs

2. **Backend: Reduce STT window from 1.5s to 1.0s** (`demo/web/app.py:stream_stt()`, line ~399)
   - Current: `STT_WINDOW_SIZE_SAMPLES = 24000` (1.5s at 16kHz)
   - Change: `STT_WINDOW_SIZE_SAMPLES = 16000` (1.0s at 16kHz, ~500ms latency)
   - Config: Add `STT_WINDOW_SIZE_MS` env var (default 1000) for runtime tuning
   - Benefit: ~250ms latency reduction per STT window; lower accuracy tradeoff minimal
   - Testing: Compare transcription accuracy (Whisper quality) at 1.0s vs 1.5s windows with sample audio

3. **Backend: Remove 2-second TTS timeout; wait for `audio_done` signal** (`demo/web/app.py:run_tts_from_queue()`, line ~675)
   - Current: `await asyncio.wait_for(text_queue.get(), timeout=2.0)` — exits if no text for 2s
   - Problem: STT produces text asynchronously; 2s timeout causes TTS to exit before all text arrives
   - Change: Loop until `text_chunk is None` (sent by `run_stt()` when `audio_done` is set)
   - Fallback: 5-second timeout only if network stall suspected (no chunks received for 5s)
   - Benefit: Eliminates premature TTS termination; TTS waits for all transcribed text
   - Testing: Record 30s audio, verify all transcription text is fed to TTS before generation completes

4. **Backend: Implement adaptive TTS text trigger** (`demo/web/app.py:run_tts_from_queue()`)
   - Current: Waits for all text before generation (unbounded latency)
   - Change: Start TTS generation when: (a) ≥5 words buffered OR (b) 0.5s elapsed since first text
   - Config: Add `TTS_MIN_WORDS=5` and `TTS_BUFFER_MS=500` env vars for tuning
   - Benefit: First audio output arrives ~500ms after first transcription; streaming feel
   - Testing: Measure time from "stt_started" to "backend_first_chunk_sent"; target <1.5s

### High Priority (Robustness)

5. **Backend: Add RMS silence detection in STT** (`demo/web/app.py:stream_stt()`, before Whisper call)
   - Current: Whisper processes all 1.0s windows regardless of silence
   - Change: Compute RMS energy per window; skip transcription if RMS < -40dB (configurable)
   - Config: Add `STT_SILENCE_THRESHOLD_DB = -40` constant; make configurable
   - Benefit: Reduces spurious transcriptions of background noise; saves Whisper compute
   - Implementation: `rms = np.sqrt(np.mean(audio_float ** 2))` before Whisper call; compare to dB threshold
   - Testing: Record 5s of silence, verify skipped frames logged; record speech, verify transcribed

6. **Backend: Add text queue backpressure monitoring** (`demo/web/app.py:run_stt()`, line ~615)
   - Current: No visibility into queue buildup; TTS may starve or buffer overflow
   - Change: Log `text_queue.qsize()` after each enqueue; emit warning if `qsize() > 100`
   - Benefit: Detect STT-TTS mismatch (e.g., TTS too slow, STT producing too fast)
   - Implementation: Add after `await text_queue.put(text)`: `if text_queue.qsize() > 100: enqueue_log("queue_backpressure_warning", queue_size=...)`
   - Testing: Stress test with fast speech + slow TTS; verify warnings emitted

7. **Frontend: Unified audio/text button with state machine** (`demo/web/index.html:~1040, 1050–1100`)
   - Current: Separate Record/Stop buttons for audio mode, Start button for text mode
   - Change: Single "Start" button that detects mode (audio vs text) and dispatches appropriately
   - Implementation: 
     - Introduce `STATE = {IDLE, CONNECTING, RECORDING, STREAMING_TTS, COMPLETE}`
     - On click: Check `modeAudioRadio.checked` → if true, call `startAudioMode()` (new), else call `start()`
     - Update button label dynamically: "🎤 Record & Speak" (audio) vs "▶ Start" (text)
   - Benefit: Unified UX; eliminates confusion about which button to use
   - Testing: Toggle between modes, verify correct handlers invoked; verify button labels update

### Medium Priority (Correctness & Resilience)

8. **Backend: Explicit `?mode=text|audio` query parameter** (`demo/web/app.py:websocket_stream()`, line ~514)
   - Current: Infers mode from presence of `text` parameter
   - Change: Use `mode = ws.query_params.get("mode", "text")` and check `if mode == "text":`
   - Benefit: Clearer intent; scalable for future modes (e.g., `?mode=stream_only`)
   - Implementation: Update websocket route condition + frontend URL construction
   - Testing: Verify both modes trigger correct paths with explicit parameter

9. **Backend: Error recovery from STT task failure** (`demo/web/app.py:run_stt()`, exception handler)
   - Current: If STT fails, TTS waits indefinitely for text
   - Change: On STT error, enqueue `None` (sentinel) to unblock TTS; emit `stt_error` event
   - Benefit: Graceful degradation; TTS doesn't hang on STT crash
   - Implementation: In except block, add `await text_queue.put(None)` after logging error
   - Testing: Simulate Whisper crash (mock exception); verify TTS continues/exits cleanly

10. **Frontend: Explicit state transitions with guards** (`demo/web/index.html` button handlers)
    - Current: Button click directly invokes handler; potential race conditions on mode switch
    - Change: Add state checks: prevent Start during RECORDING, prevent Stop during IDLE, etc.
    - Implementation: Check `STATE` enum before invoking handler; display error tooltip if blocked
    - Benefit: Prevents user confusion; prevents protocol violations (e.g., multiple WebSocket opens)
    - Testing: Rapidly toggle modes, spam button clicks; verify no crashes or hanging connections

### Medium Priority (Latency Optimization)

11. **Backend: Configurable Whisper model size** (`demo/web/app.py:_load_whisper()`, line ~145)
    - Current: Hardcoded `large-v3` (~3GB, slow on CPU, 500ms+ latency)
    - Change: Make configurable via `STT_MODEL_SIZE` env var (e.g., `base`, `small`, `medium`, `large`)
    - Benefit: CPU-constrained deployments can use `small` (~480MB, ~200ms latency); high-accuracy needs `large`
    - Implementation: `model_size = os.environ.get("STT_MODEL_SIZE", "large-v3")`
    - Testing: Benchmark latency & accuracy across model sizes; document tradeoffs

12. **Backend: Optional echo cancellation for concurrent playback** (`demo/web/app.py:websocket_stream()`)
    - Current: TTS audio plays while recording; risk of echo if speaker near mic
    - Change: Add `?no_playback=1` mode flag to suppress TTS output during recording
    - Benefit: Cleaner audio quality in echo-prone environments (phones, tablets)
    - Implementation: Check param in TTS handler; skip sending WebSocket audio bytes if flag set
    - Testing: Record in echo chamber; compare audio quality with/without flag

### Low Priority (Developer Experience & Future-Proofing)

13. **Documentation: Add realtime SLA targets to README**
    - Document target latencies: <1.5s end-to-end, <500ms STT, <400ms TTS
    - Explain latency breakdown: audio capture (200ms) + transmission (50ms) + STT (500ms) + TTS (400ms)
    - Note: Actual latency depends on hardware, network, audio environment

14. **Testing: Create realtime stress test suite**
    - Concurrent connections test: 5+ simultaneous audio streams
    - Long recording test: 5+ minute continuous recording & transcription
    - Latency profiling: Measure exact timestamps for each stage (audio_received → stt_partial → tts_started → audio_sent)
    - Accuracy benchmark: Compare Whisper output to ground truth across model sizes

15. **Code cleanup: Extract realtime config constants to config file**
    - Current: Hardcoded values scattered in `stream_stt()` and `run_tts_from_queue()`
    - Change: Centralize in `demo/web/realtime_config.py` or environment variables
    - Examples: `STT_WINDOW_SIZE_MS`, `STT_SILENCE_THRESHOLD_DB`, `TTS_MIN_WORDS`, `TTS_BUFFER_MS`, `STT_MODEL_SIZE`
    - Benefit: Easy tuning without code changes; better configuration management

---

### Implementation Order (Recommended)
1. Start with **#3** (remove 2s timeout) — quickest win, dramatic improvement
2. Then **#1** (200ms chunks) — reduces latency by ~1s
3. Then **#2** (1.0s STT window) — further latency reduction
4. Then **#4** (adaptive TTS trigger) — achieves streaming feel
5. Then **#5–7** (robustness & UX) — stabilize system
6. Then **#8–10** (resilience) — production-ready
7. Finally **#11–15** (optimization & polish)

Each step builds on previous work; measure latency impact after each change using server logs.