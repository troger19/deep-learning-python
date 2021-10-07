# import PyAudio
# conda install PyAudio
# conda install ffmpeg

from pydub import AudioSegment
from pydub.playback import play

mp3_file = AudioSegment.from_file(file="sounds/Bach.mp3", format="mp3")

play(mp3_file)