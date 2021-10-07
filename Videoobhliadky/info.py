
from pydub import AudioSegment

wav_file = AudioSegment.from_file(file="sounds/Bach.mp3", format="mp3")

# data type fo the file
print(type(wav_file))
# OUTPUT: <class 'pydub.audio_segment.AudioSegment'>

#  To find frame rate of song/file
print(wav_file.frame_rate)
# OUTPUT: 22050

# To know about channels of file
print(wav_file.channels)
# OUTPUT: 1

# Find the number of bytes per sample
print(wav_file.sample_width)
# OUTPUT : 2


# Find Maximum amplitude
print(wav_file.max)
# OUTPUT 17106

# To know length of audio file
print(len(wav_file))
# OUTPUT 60000

'''
We can change the attrinbutes of file by 
changeed_audio_segment = audio_segment.set_ATTRIBUTENAME(x) 
'''
wav_file_new = wav_file.set_frame_rate(50)
print(wav_file_new.frame_rate)