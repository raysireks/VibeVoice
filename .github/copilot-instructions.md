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

