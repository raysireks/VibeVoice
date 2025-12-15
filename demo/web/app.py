import datetime
import builtins
import asyncio
import json
import os
import threading
import traceback
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Callable, Dict, Iterator, Optional, Tuple, cast

import numpy as np
import torch
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketDisconnect, WebSocketState

from vibevoice.modular.modeling_vibevoice_streaming_inference import (
    VibeVoiceStreamingForConditionalGenerationInference,
)
from vibevoice.processor.vibevoice_streaming_processor import (
    VibeVoiceStreamingProcessor,
)
from vibevoice.modular.streamer import AudioStreamer

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

import copy

BASE = Path(__file__).parent
SAMPLE_RATE = 24_000


def get_timestamp():
    timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc
    ).astimezone(
        datetime.timezone(datetime.timedelta(hours=8))
    ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return timestamp

class StreamingTTSService:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        inference_steps: int = 5,
    ) -> None:
        # Keep model_path as string for HuggingFace repo IDs (Path() converts / to \ on Windows)
        self.model_path = model_path
        self.inference_steps = inference_steps
        self.sample_rate = SAMPLE_RATE

        self.processor: Optional[VibeVoiceStreamingProcessor] = None
        self.model: Optional[VibeVoiceStreamingForConditionalGenerationInference] = None
        self.voice_presets: Dict[str, Path] = {}
        self.default_voice_key: Optional[str] = None
        self._voice_cache: Dict[str, Tuple[object, Path, str]] = {}
        
        # STT (Faster Whisper) components
        self.whisper_model: Optional[Any] = None  # WhisperModel instance
        self.stt_device = device  # Use same device as TTS

        if device == "mpx":
            print("Note: device 'mpx' detected, treating it as 'mps'.")
            device = "mps"        
        if device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS not available. Falling back to CPU.")
            device = "cpu"
        self.device = device
        self._torch_device = torch.device(device)

    def load(self) -> None:
        print(f"[startup] Loading processor from {self.model_path}")
        self.processor = VibeVoiceStreamingProcessor.from_pretrained(self.model_path)

        
        # Decide dtype & attention
        if self.device == "mps":
            load_dtype = torch.float32
            device_map = None
            attn_impl_primary = "sdpa"
        elif self.device == "cuda":
            load_dtype = torch.bfloat16
            device_map = 'cuda'
            attn_impl_primary = "flash_attention_2"
        else:
            load_dtype = torch.float32
            device_map = 'cpu'
            attn_impl_primary = "sdpa"
        print(f"Using device: {device_map}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        # Load model
        try:
            self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                self.model_path,
                torch_dtype=load_dtype,
                device_map=device_map,
                attn_implementation=attn_impl_primary,
            )
            
            if self.device == "mps":
                self.model.to("mps")
        except Exception as e:
            if attn_impl_primary == 'flash_attention_2':
                print("Error loading the model. Trying to use SDPA. However, note that only flash_attention_2 has been fully tested, and using SDPA may result in lower audio quality.")
                
                self.model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
                    self.model_path,
                    torch_dtype=load_dtype,
                    device_map=self.device,
                    attn_implementation='sdpa',
                )
                print("Load model with SDPA successfully ")
            else:
                raise e

        self.model.eval()

        self.model.model.noise_scheduler = self.model.model.noise_scheduler.from_config(
            self.model.model.noise_scheduler.config,
            algorithm_type="sde-dpmsolver++",
            beta_schedule="squaredcos_cap_v2",
        )
        self.model.set_ddpm_inference_steps(num_steps=self.inference_steps)

        self.voice_presets = self._load_voice_presets()
        preset_name = os.environ.get("VOICE_PRESET")
        self.default_voice_key = self._determine_voice_key(preset_name)
        self._ensure_voice_cached(self.default_voice_key)
        
        # Lazy-load Whisper model for STT
        self._load_whisper()

    def _load_whisper(self) -> None:
        """Lazy-load Faster Whisper model for speech-to-text."""
        if WhisperModel is None:
            print("[startup] Faster Whisper not installed. STT features unavailable.")
            return
        
        try:
            print(f"[startup] Loading Faster Whisper model (large-v3) on device {self.stt_device}")
            # Map TTS device names to Whisper compatible device strings
            whisper_device = "cuda" if self.stt_device == "cuda" else "cpu"
            compute_type = "float16" if whisper_device == "cuda" else "int8"
            
            self.whisper_model = WhisperModel(
                model_size_or_path="large-v3",
                device=whisper_device,
                compute_type=compute_type,
            )
            print("[startup] Faster Whisper model loaded successfully")
        except Exception as e:
            print(f"[startup] Failed to load Faster Whisper model: {e}")
            self.whisper_model = None

    def _load_voice_presets(self) -> Dict[str, Path]:
        voices_dir = BASE.parent / "voices" / "streaming_model"
        if not voices_dir.exists():
            raise RuntimeError(f"Voices directory not found: {voices_dir}")

        presets: Dict[str, Path] = {}
        for pt_path in voices_dir.glob("*.pt"):
            presets[pt_path.stem] = pt_path

        if not presets:
            raise RuntimeError(f"No voice preset (.pt) files found in {voices_dir}")

        print(f"[startup] Found {len(presets)} voice presets")
        return dict(sorted(presets.items()))

    def _determine_voice_key(self, name: Optional[str]) -> str:
        if name and name in self.voice_presets:
            return name

        default_key = "en-WHTest_man"
        if default_key in self.voice_presets:
            return default_key

        first_key = next(iter(self.voice_presets))
        print(f"[startup] Using fallback voice preset: {first_key}")
        return first_key

    def _ensure_voice_cached(self, key: str) -> Tuple[object, Path, str]:
        if key not in self.voice_presets:
            raise RuntimeError(f"Voice preset {key!r} not found")

        if key not in self._voice_cache:
            preset_path = self.voice_presets[key]
            print(f"[startup] Loading voice preset {key} from {preset_path}")
            print(f"[startup] Loading prefilled prompt from {preset_path}")
            prefilled_outputs = torch.load(
                preset_path,
                map_location=self._torch_device,
                weights_only=False,
            )
            self._voice_cache[key] = prefilled_outputs

        return self._voice_cache[key]

    def _get_voice_resources(self, requested_key: Optional[str]) -> Tuple[str, object, Path, str]:
        key = requested_key if requested_key and requested_key in self.voice_presets else self.default_voice_key
        if key is None:
            key = next(iter(self.voice_presets))
            self.default_voice_key = key

        prefilled_outputs = self._ensure_voice_cached(key)
        return key, prefilled_outputs

    def _prepare_inputs(self, text: str, prefilled_outputs: object):
        if not self.processor or not self.model:
            raise RuntimeError("StreamingTTSService not initialized")

        processor_kwargs = {
            "text": text.strip(),
            "cached_prompt": prefilled_outputs,
            "padding": True,
            "return_tensors": "pt",
            "return_attention_mask": True,
        }

        processed = self.processor.process_input_with_cached_prompt(**processor_kwargs)

        prepared = {
            key: value.to(self._torch_device) if hasattr(value, "to") else value
            for key, value in processed.items()
        }
        return prepared

    def _run_generation(
        self,
        inputs,
        audio_streamer: AudioStreamer,
        errors,
        cfg_scale: float,
        do_sample: bool,
        temperature: float,
        top_p: float,
        refresh_negative: bool,
        prefilled_outputs,
        stop_event: threading.Event,
    ) -> None:
        try:
            self.model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=cfg_scale,
                tokenizer=self.processor.tokenizer,
                generation_config={
                    "do_sample": do_sample,
                    "temperature": temperature if do_sample else 1.0,
                    "top_p": top_p if do_sample else 1.0,
                },
                audio_streamer=audio_streamer,
                stop_check_fn=stop_event.is_set,
                verbose=False,
                refresh_negative=refresh_negative,
                all_prefilled_outputs=copy.deepcopy(prefilled_outputs),
            )
        except Exception as exc:  # pragma: no cover - diagnostic logging
            errors.append(exc)
            traceback.print_exc()
            audio_streamer.end()

    def stream(
        self,
        text: str,
        cfg_scale: float = 1.5,
        do_sample: bool = False,
        temperature: float = 0.9,
        top_p: float = 0.9,
        refresh_negative: bool = True,
        inference_steps: Optional[int] = None,
        voice_key: Optional[str] = None,
        log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        stop_event: Optional[threading.Event] = None,
    ) -> Iterator[np.ndarray]:
        if not text.strip():
            return
        text = text.replace("’", "'")
        selected_voice, prefilled_outputs = self._get_voice_resources(voice_key)

        def emit(event: str, **payload: Any) -> None:
            if log_callback:
                try:
                    log_callback(event, **payload)
                except Exception as exc:
                    print(f"[log_callback] Error while emitting {event}: {exc}")

        steps_to_use = self.inference_steps
        if inference_steps is not None:
            try:
                parsed_steps = int(inference_steps)
                if parsed_steps > 0:
                    steps_to_use = parsed_steps
            except (TypeError, ValueError):
                pass
        if self.model:
            self.model.set_ddpm_inference_steps(num_steps=steps_to_use)
        self.inference_steps = steps_to_use

        inputs = self._prepare_inputs(text, prefilled_outputs)
        audio_streamer = AudioStreamer(batch_size=1, stop_signal=None, timeout=None)
        errors: list = []
        stop_signal = stop_event or threading.Event()

        thread = threading.Thread(
            target=self._run_generation,
            kwargs={
                "inputs": inputs,
                "audio_streamer": audio_streamer,
                "errors": errors,
                "cfg_scale": cfg_scale,
                "do_sample": do_sample,
                "temperature": temperature,
                "top_p": top_p,
                "refresh_negative": refresh_negative,
                "prefilled_outputs": prefilled_outputs,
                "stop_event": stop_signal,
            },
            daemon=True,
        )
        thread.start()

        generated_samples = 0

        try:
            stream = audio_streamer.get_stream(0)
            for audio_chunk in stream:
                if torch.is_tensor(audio_chunk):
                    audio_chunk = audio_chunk.detach().cpu().to(torch.float32).numpy()
                else:
                    audio_chunk = np.asarray(audio_chunk, dtype=np.float32)

                if audio_chunk.ndim > 1:
                    audio_chunk = audio_chunk.reshape(-1)

                peak = np.max(np.abs(audio_chunk)) if audio_chunk.size else 0.0
                if peak > 1.0:
                    audio_chunk = audio_chunk / peak

                generated_samples += int(audio_chunk.size)
                emit(
                    "model_progress",
                    generated_sec=generated_samples / self.sample_rate,
                    chunk_sec=audio_chunk.size / self.sample_rate,
                )

                chunk_to_yield = audio_chunk.astype(np.float32, copy=False)

                yield chunk_to_yield
        finally:
            stop_signal.set()
            audio_streamer.end()
            thread.join()
            if errors:
                emit("generation_error", message=str(errors[0]))
                raise errors[0]

    def chunk_to_pcm16(self, chunk: np.ndarray) -> bytes:
        chunk = np.clip(chunk, -1.0, 1.0)
        pcm = (chunk * 32767.0).astype(np.int16)
        return pcm.tobytes()

    async def stream_stt(
        self,
        audio_generator: Any,  # async iterator of bytes
        log_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """
        Stream speech-to-text using Faster Whisper with partial transcriptions.
        
        Audio is expected as PCM16 bytes at 16 kHz sample rate.
        Yields partial transcription dictionaries as audio is buffered and processed.
        
        Args:
            audio_generator: Iterator yielding PCM16 bytes (16 kHz)
            log_callback: Optional callback for logging events
            
        Yields:
            Dicts with keys: {'text': str, 'is_final': bool, 'timestamp_ms': float}
        """
        if not self.whisper_model:
            raise RuntimeError("Faster Whisper model not loaded")
        
        def emit(event: str, **payload: Any) -> None:
            if log_callback:
                try:
                    log_callback(event, **payload)
                except Exception as exc:
                    print(f"[stt_callback] Error while emitting {event}: {exc}")
        
        # Audio buffer: accumulate 1.0 second at 16 kHz = 16,000 samples (reduced from 1.5s for lower latency)
        STT_WINDOW_SIZE_MS = int(os.environ.get("STT_WINDOW_SIZE_MS", "1000"))
        STT_WINDOW_SIZE_SAMPLES = int(16000 * STT_WINDOW_SIZE_MS / 1000)  # Dynamic window size
        STT_SAMPLE_RATE = 16000
        STT_SILENCE_THRESHOLD_DB = float(os.environ.get("STT_SILENCE_THRESHOLD_DB", "-40"))
        
        def is_silence(audio_float: np.ndarray) -> bool:
            """Check if audio window is below silence threshold using RMS energy."""
            if audio_float.size == 0:
                return True
            rms = np.sqrt(np.mean(audio_float ** 2))
            # Convert RMS to dB (20 * log10(rms))
            db = 20 * np.log10(rms + 1e-10)  # Add epsilon to avoid log(0)
            return db < STT_SILENCE_THRESHOLD_DB
        
        audio_buffer = np.array([], dtype=np.int16)
        total_samples_processed = 0
        
        try:
            async for audio_chunk_bytes in audio_generator:
                # Convert PCM16 bytes to int16 numpy array
                chunk = np.frombuffer(audio_chunk_bytes, dtype=np.int16)
                audio_buffer = np.concatenate([audio_buffer, chunk])
                
                # When buffer reaches target size, transcribe
                while len(audio_buffer) >= STT_WINDOW_SIZE_SAMPLES:
                    window = audio_buffer[:STT_WINDOW_SIZE_SAMPLES]
                    audio_buffer = audio_buffer[STT_WINDOW_SIZE_SAMPLES:]
                    
                    # Convert to float32 in range [-1, 1] for Whisper
                    audio_float = window.astype(np.float32) / 32768.0
                    
                    # Skip silent frames (reduces spurious transcriptions and saves compute)
                    if is_silence(audio_float):
                        total_samples_processed += STT_WINDOW_SIZE_SAMPLES
                        print(f"[stt] Skipped silent frame at t={total_samples_processed/STT_SAMPLE_RATE:.2f}s")
                        emit("transcription_skipped_silence", timestamp_s=total_samples_processed/STT_SAMPLE_RATE)
                        continue
                    
                    # Transcribe this window
                    try:
                        segments, info = self.whisper_model.transcribe(
                            audio_float,
                            language="en",
                            beam_size=5,
                            vad_filter=False,  # Explicit control - no VAD
                            condition_on_previous_text=False,
                        )
                        
                        transcribed_text = ""
                        for segment in segments:
                            transcribed_text += segment.text + " "
                        
                        transcribed_text = transcribed_text.strip()
                        total_samples_processed += STT_WINDOW_SIZE_SAMPLES
                        if transcribed_text:
                            timestamp_ms = (total_samples_processed / STT_SAMPLE_RATE) * 1000
                            
                            result = {
                                "text": transcribed_text,
                                "is_final": False,
                                "timestamp_ms": timestamp_ms,
                            }
                            print(f"[stt_partial] t={timestamp_ms/1000:.2f}s text='{transcribed_text}'")
                            emit("transcription_partial", **result)
                            yield result
                    except Exception as e:
                        print(f"[stt] Transcription error: {e}")
                        emit("transcription_error", message=str(e))
        finally:
            # Handle remaining audio in buffer when stream ends
            if len(audio_buffer) > 0:
                audio_float = audio_buffer.astype(np.float32) / 32768.0
                # Skip final frame if it's silence
                if not is_silence(audio_float):
                    try:
                        segments, info = self.whisper_model.transcribe(
                            audio_float,
                            language="en",
                            beam_size=5,
                            vad_filter=False,
                            condition_on_previous_text=False,
                        )
                        
                        transcribed_text = ""
                        for segment in segments:
                            transcribed_text += segment.text + " "
                        
                        transcribed_text = transcribed_text.strip()
                        if transcribed_text:
                            total_samples_processed += len(audio_buffer)
                            timestamp_ms = (total_samples_processed / STT_SAMPLE_RATE) * 1000
                            
                            result = {
                                "text": transcribed_text,
                                "is_final": True,
                                "timestamp_ms": timestamp_ms,
                            }
                            print(f"[stt_final] t={timestamp_ms/1000:.2f}s text='{transcribed_text}'")
                            emit("transcription_final", **result)
                            yield result
                    except Exception as e:
                        print(f"[stt] Final transcription error: {e}")
                        emit("transcription_error", message=str(e))




# FastAPI application instance
app = FastAPI()


@app.on_event("startup")
async def _startup() -> None:
    model_path = os.environ.get("MODEL_PATH")
    if not model_path:
        raise RuntimeError("MODEL_PATH not set in environment")

    device = os.environ.get("MODEL_DEVICE", "cuda")
    
    service = StreamingTTSService(
        model_path=model_path,
        device=device
    )
    service.load()

    app.state.tts_service = service
    app.state.model_path = model_path
    app.state.device = device
    app.state.websocket_lock = asyncio.Lock()
    print("[startup] Model ready.")


def streaming_tts(text: str, **kwargs) -> Iterator[np.ndarray]:
    service: StreamingTTSService = app.state.tts_service
    yield from service.stream(text, **kwargs)

@app.websocket("/stream")
async def websocket_stream(ws: WebSocket) -> None:
    await ws.accept()
    text = ws.query_params.get("text", "")
    voice_param = ws.query_params.get("voice")
    
    service: StreamingTTSService = app.state.tts_service
    tts_lock: asyncio.Lock = app.state.websocket_lock
    
    # Separate lock for STT to allow concurrent streams
    stt_mode = False
    audio_buffer = []  # retained for compatibility, no longer used for streaming
    audio_chunk_queue: "asyncio.Queue[Optional[bytes]]" = asyncio.Queue()
    audio_done = asyncio.Event()  # signals client sent audio_end / websocket closed
    stt_task = None
    tts_task = None
    text_queue = asyncio.Queue()
    
    log_queue: "Queue[Dict[str, Any]]" = Queue()
    
    def enqueue_log(event: str, **data: Any) -> None:
        log_queue.put({"event": event, "data": data})
    
    async def flush_logs() -> None:
        while True:
            try:
                entry = log_queue.get_nowait()
            except Empty:
                break
            message = {
                "type": "log",
                "event": entry.get("event"),
                "data": entry.get("data", {}),
                "timestamp": get_timestamp(),
            }
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                break
    
    async def handle_audio_stream() -> None:
        """Handle incoming audio chunks and feed to STT in near-real-time."""
        try:
            print("[audio] Starting audio reception handler")
            while ws.client_state == WebSocketState.CONNECTED:
                try:
                    # Receive audio chunk with timeout
                    message = await asyncio.wait_for(ws.receive(), timeout=30.0)
                    
                    chunk = message.get("bytes")
                    if chunk is not None:
                        await audio_chunk_queue.put(chunk)
                        print(f"[audio] Received {len(chunk)} bytes (queue size: {audio_chunk_queue.qsize()})")
                        continue

                    text_payload = message.get("text")
                    if text_payload is not None:
                        try:
                            data = json.loads(text_payload or "{}")
                            msg_type = data.get("type")
                            
                            if msg_type == "audio_end":
                                print("[audio] Client signaled end of audio stream")
                                enqueue_log("audio_stream_ended")
                                break
                            elif msg_type == "voice_change":
                                new_voice = data.get("voice")
                                enqueue_log("voice_changed", voice=new_voice)
                        except json.JSONDecodeError:
                            pass
                        continue

                except asyncio.TimeoutError:
                    print("[audio] Timeout waiting for audio data")
                    enqueue_log("audio_timeout")
                    break
                except Exception as e:
                    print(f"[audio] Error receiving data: {e}")
                    break
        except Exception as e:
            print(f"[audio_stream] Error: {e}")
        finally:
            print("[audio] Reception complete. Signaling audio_done.")
            audio_done.set()
            # Sentinel None to unblock consumers
            await audio_chunk_queue.put(None)
            enqueue_log("audio_reception_complete")
    
    async def run_stt() -> None:
        """Run STT on streaming audio and feed to TTS with backpressure monitoring."""
        try:
            enqueue_log("stt_started")
            transcribed_text = ""

            async def audio_gen():
                # Consume from queue as chunks arrive; stop on sentinel None
                while True:
                    chunk = await audio_chunk_queue.get()
                    if chunk is None:
                        break
                    yield chunk

            stt_gen = service.stream_stt(
                audio_gen(),
                log_callback=enqueue_log
            )

            async for result in stt_gen:
                text = result.get("text", "").strip()
                is_final = result.get("is_final", False)

                if text:
                    transcribed_text += text + " "
                    # Queue text for TTS processing
                    await text_queue.put(text)
                    queue_size = text_queue.qsize()
                    # Log backpressure warnings if queue is backing up
                    if queue_size > 100:
                        enqueue_log("queue_backpressure_warning", queue_size=queue_size, text_snippet=text[:50])
                    enqueue_log("transcription_update", text=transcribed_text.strip(), is_final=is_final, queue_size=queue_size)

            # Signal that STT is complete
            await text_queue.put(None)
            enqueue_log("stt_completed")

        except Exception as e:
            print(f"[stt_task] Error: {e}")
            enqueue_log("stt_error", message=str(e))
            # Signal TTS to exit cleanly on STT failure
            await text_queue.put(None)
    
    async def run_tts_from_queue() -> None:
        """Read text from queue and generate TTS."""
        try:
            cfg_param = ws.query_params.get("cfg")
            steps_param = ws.query_params.get("steps")
            
            try:
                cfg_scale = float(cfg_param) if cfg_param else 1.5
            except (ValueError, TypeError):
                cfg_scale = 1.5
            if cfg_scale <= 0:
                cfg_scale = 1.5
            
            try:
                inference_steps = int(steps_param) if steps_param else None
                if inference_steps is not None and inference_steps <= 0:
                    inference_steps = None
            except (ValueError, TypeError):
                inference_steps = None
            
            # Acquire TTS lock only when generating
            async with tts_lock:
                enqueue_log("tts_started")
                
                accumulated_text = ""
                stop_signal = threading.Event()
                tts_min_words = int(os.environ.get("TTS_MIN_WORDS", "5"))
                tts_buffer_ms = int(os.environ.get("TTS_BUFFER_MS", "500"))
                
                # Continuously process text chunks as they arrive from STT
                first_ws_send_logged = False
                stt_complete = False  # Track if STT has signaled completion
                
                while True:
                    # Accumulate text until adaptive trigger threshold
                    tts_buffer_start_time = asyncio.get_event_loop().time()
                    chunk_batch = ""
                    
                    while True:
                        try:
                            # Wait for text from STT queue (no timeout - wait indefinitely until STT completes)
                            text_chunk = await text_queue.get()
                        except asyncio.TimeoutError:
                            # This should not happen with no timeout, but handle gracefully
                            if chunk_batch.strip():
                                print("[tts_task] Timeout waiting for more text, processing buffered text")
                                enqueue_log("tts_timeout_processing_buffer", text_length=len(chunk_batch))
                                break
                            else:
                                # Should never reach here - no timeout set
                                print("[tts_task] Unexpected timeout")
                                enqueue_log("tts_unexpected_timeout")
                                enqueue_log("tts_completed")
                                return
                        
                        if text_chunk is None:
                            # STT stream complete - process final accumulated text and exit
                            print(f"[tts_task] STT complete signal received. Final batch: '{chunk_batch}'")
                            stt_complete = True
                            if chunk_batch.strip():
                                break  # Process final batch
                            else:
                                enqueue_log("tts_completed")
                                return
                        
                        chunk_batch += text_chunk + " "
                        
                        # Check if we should start TTS generation now (adaptive trigger)
                        word_count = len(chunk_batch.split())
                        elapsed_ms = (asyncio.get_event_loop().time() - tts_buffer_start_time) * 1000
                        
                        if word_count >= tts_min_words or elapsed_ms >= tts_buffer_ms:
                            print(f"[tts_task] TTS trigger: words={word_count}, elapsed_ms={elapsed_ms:.0f}")
                            enqueue_log("tts_buffer_trigger", words=word_count, elapsed_ms=elapsed_ms, threshold_words=tts_min_words, threshold_ms=tts_buffer_ms)
                            break  # Start TTS generation with this batch
                    
                    # Generate TTS for this batch
                    if chunk_batch.strip():
                        accumulated_text += chunk_batch
                        enqueue_log(
                            "tts_generating",
                            text_length=len(chunk_batch),
                            word_count=len(chunk_batch.split()),
                            total_text_length=len(accumulated_text),
                            cfg_scale=cfg_scale,
                            inference_steps=inference_steps,
                        )
                        
                        iterator = streaming_tts(
                            chunk_batch,
                            cfg_scale=cfg_scale,
                            inference_steps=inference_steps,
                            voice_key=voice_param,
                            log_callback=enqueue_log,
                            stop_event=stop_signal,
                        )
                        sentinel = object()
                        
                        try:
                            while ws.client_state == WebSocketState.CONNECTED:
                                await flush_logs()
                                chunk = await asyncio.to_thread(next, iterator, sentinel)
                                if chunk is sentinel:
                                    break
                                chunk = cast(np.ndarray, chunk)
                                payload = service.chunk_to_pcm16(chunk)
                                await ws.send_bytes(payload)
                                if not first_ws_send_logged:
                                    first_ws_send_logged = True
                                    enqueue_log("backend_first_chunk_sent")
                                await flush_logs()
                        except WebSocketDisconnect:
                            print("Client disconnected during TTS")
                            enqueue_log("client_disconnected")
                            stop_signal.set()
                            return
                        finally:
                            try:
                                iterator_close = getattr(iterator, "close", None)
                                if callable(iterator_close):
                                    iterator_close()
                            except Exception:
                                pass
                    
                    # Check if we should continue (look for more text or exit)
                    if stt_complete:  # STT signaled completion - exit after processing final batch
                        print("[tts_task] STT complete - exiting TTS loop")
                        break
                
                enqueue_log("tts_completed")
        
        except Exception as e:
            print(f"[tts_task] Error: {e}")
            enqueue_log("tts_error", message=str(e))
    
    print(f"Client connected, text={text!r}, voice={voice_param!r}")
    
    try:
        # Determine operation mode based on explicit mode parameter
        mode = ws.query_params.get("mode", "text")  # Default to text mode for backward compatibility
        if mode == "text":
            # Text input mode (existing behavior)
            print("[mode] Text-to-speech mode")
            
            cfg_param = ws.query_params.get("cfg")
            steps_param = ws.query_params.get("steps")
            
            try:
                cfg_scale = float(cfg_param) if cfg_param is not None else 1.5
            except ValueError:
                cfg_scale = 1.5
            if cfg_scale <= 0:
                cfg_scale = 1.5
            try:
                inference_steps = int(steps_param) if steps_param is not None else None
                if inference_steps is not None and inference_steps <= 0:
                    inference_steps = None
            except ValueError:
                inference_steps = None
            
            # Use TTS lock for text mode
            if tts_lock.locked():
                busy_message = {
                    "type": "log",
                    "event": "backend_busy",
                    "data": {"message": "Please wait for the other requests to complete."},
                    "timestamp": get_timestamp(),
                }
                print("Service busy")
                try:
                    await ws.send_text(json.dumps(busy_message))
                except Exception:
                    pass
                await ws.close(code=1013, reason="Service busy")
                return
            
            async with tts_lock:
                log_queue: "Queue[Dict[str, Any]]" = Queue()
                
                def enqueue_log(event: str, **data: Any) -> None:
                    log_queue.put({"event": event, "data": data})
                
                enqueue_log(
                    "backend_request_received",
                    text_length=len(text or ""),
                    cfg_scale=cfg_scale,
                    inference_steps=inference_steps,
                    voice=voice_param,
                )
                
                stop_signal = threading.Event()
                
                iterator = streaming_tts(
                    text,
                    cfg_scale=cfg_scale,
                    inference_steps=inference_steps,
                    voice_key=voice_param,
                    log_callback=enqueue_log,
                    stop_event=stop_signal,
                )
                sentinel = object()
                first_ws_send_logged = False
                
                await flush_logs()
                
                try:
                    while ws.client_state == WebSocketState.CONNECTED:
                        await flush_logs()
                        chunk = await asyncio.to_thread(next, iterator, sentinel)
                        if chunk is sentinel:
                            break
                        chunk = cast(np.ndarray, chunk)
                        payload = service.chunk_to_pcm16(chunk)
                        await ws.send_bytes(payload)
                        if not first_ws_send_logged:
                            first_ws_send_logged = True
                            enqueue_log("backend_first_chunk_sent")
                        await flush_logs()
                except WebSocketDisconnect:
                    print("Client disconnected (WebSocketDisconnect)")
                    enqueue_log("client_disconnected")
                    stop_signal.set()
                finally:
                    stop_signal.set()
                    enqueue_log("backend_stream_complete")
                    await flush_logs()
                    try:
                        iterator_close = getattr(iterator, "close", None)
                        if callable(iterator_close):
                            iterator_close()
                    except Exception:
                        pass
                    while not log_queue.empty():
                        try:
                            log_queue.get_nowait()
                        except Empty:
                            break
        else:
            # Audio input mode (STT + TTS)
            print("[mode] Audio-to-speech mode (STT + TTS)")
            stt_mode = True
            
            enqueue_log("ready_for_audio")
            await flush_logs()
            
            # Start all tasks concurrently
            # STT will wait for audio_ready event before processing
            audio_task = asyncio.create_task(handle_audio_stream())
            stt_task = asyncio.create_task(run_stt())
            tts_task = asyncio.create_task(run_tts_from_queue())
           
            # Wait for all tasks to complete
            try:
                await asyncio.gather(audio_task, stt_task, tts_task)
            except asyncio.CancelledError:
                pass
            finally:
                enqueue_log("audio_to_speech_complete")
                await flush_logs()
    
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"[websocket] Unexpected error: {e}")
        enqueue_log("websocket_error", message=str(e))
        await flush_logs()
    finally:
        try:
            if ws.client_state == WebSocketState.CONNECTED:
                await ws.close()
        except Exception:
            pass
        print("WS handler exit")


@app.get("/")
def index():
    return FileResponse(BASE / "index.html")


@app.get("/config")
def get_config():
    service: StreamingTTSService = app.state.tts_service
    voices = sorted(service.voice_presets.keys())
    return {
        "voices": voices,
        "default_voice": service.default_voice_key,
    }

