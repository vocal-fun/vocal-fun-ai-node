from faster_whisper import WhisperModel
import time

def transcribe_audio(audio_path):
    start_time = time.time()

    print(f"Starting transcription...")
    segments, info = model.transcribe(
        audio_path,
        beam_size=1,          # Reduce beam size for faster inference
    )

    transcribed_text = " ".join([segment.text for segment in segments])

    end_time = time.time()
    time_taken = end_time - start_time

    print(f"Transcription completed in {time_taken:.2f} seconds")
    return transcribed_text

if __name__ == "__main__":
    # Use CUDA if available, otherwise fall back to CPU
    print("Loading Whisper tiny.en model...")
    model = WhisperModel(
        "tiny.en",
        device="cuda",        # Use GPU if available, will fall back to cpu if not
        compute_type="int8",  # Use int8 quantization for faster processing
        download_root="./models"
    )

    audio_file = "xtts_streaming.wav"

    try:
        print("\nRunning 5 transcriptions:\n")
        for i in range(5):
            print(f"\nRun #{i+1}")
            transcribed_text = transcribe_audio(audio_file)
            print("Transcribed Text:")
            print(transcribed_text)

    except Exception as e:
        print(f"An error occurred: {str(e)}")