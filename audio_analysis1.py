import librosa
import numpy as np
import subprocess
import tempfile
import os
import warnings

class AudioAnalyzer:
    def __init__(self, siren_freq_range=(500, 1500), threshold=0.5):
        self.siren_freq_range = siren_freq_range
        self.threshold = threshold
    
    def extract_audio(self, video_path):
        # Create a temporary WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False).name
        try:
            # Run ffmpeg to extract audio
            result = subprocess.run(
                [
                    "ffmpeg", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                    "-ar", "44100", "-ac", "1", "-y", temp_wav
                ],
                capture_output=True, text=True, check=True
            )
            # Check if the file was created and has content
            if not os.path.exists(temp_wav) or os.path.getsize(temp_wav) == 0:
                warnings.warn(f"Audio extraction failed for {video_path}. Empty or invalid audio file.")
                return None
            return temp_wav
        except subprocess.CalledProcessError as e:
            warnings.warn(f"ffmpeg failed: {e.stderr}")
            return None
        except Exception as e:
            warnings.warn(f"Error extracting audio: {str(e)}")
            return None
    
    def detect_siren(self, audio_path):
        if audio_path is None or not os.path.exists(audio_path):
            warnings.warn("No valid audio file provided for siren detection.")
            return False
        
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=44100)
            if len(y) == 0:
                warnings.warn("Audio file is empty.")
                return False
            
            # Compute spectrogram
            S = np.abs(librosa.stft(y))
            freqs = librosa.fft_frequencies(sr=sr)
            
            # Check for siren frequencies
            siren_mask = (freqs >= self.siren_freq_range[0]) & (freqs <= self.siren_freq_range[1])
            siren_power = np.mean(S[siren_mask, :], axis=0)
            avg_power = np.mean(siren_power)
            
            return avg_power > self.threshold
        except Exception as e:
            warnings.warn(f"Error in siren detection: {str(e)}")
            return False
    
    def cleanup(self, audio_path):
        if audio_path and os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except Exception as e:
                warnings.warn(f"Failed to clean up audio file: {str(e)}")