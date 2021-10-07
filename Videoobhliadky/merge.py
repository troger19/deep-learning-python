from pydub import AudioSegment

wav_file_1 = AudioSegment.from_file("sounds/Bach.mp3")
wav_file_2 = AudioSegment.from_file("sounds/sample-3s.mp3")

# Combine the two audio files
# wav_file_3 = wav_file_1 + wav_file_2

# play the sound
# play(wav_file_3)
#
# sound1 6 dB louder
louder = wav_file_1+5

# Overlay sound2 over sound1 at position 0  (use louder instead of sound1 to use the louder version)
overlay = louder.overlay(wav_file_2, position=0)

# play(overlay)
file_handle = overlay.export("sounds/output.mp3", format="mp3")