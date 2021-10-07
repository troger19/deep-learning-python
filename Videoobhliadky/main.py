import base64
import flask
from flask import request, jsonify
from pydub import AudioSegment
from pydub.playback import play

app = flask.Flask(__name__)


@app.route('/api/play', methods=['POST'])
def play_song():
    try:
        sound1 = request.get_json(force=True)['sound1']
        sound2 = request.get_json(force=True)['sound2']
        with open("sounds/sound1.mp3", "wb") as fh:
            fh.write(base64.b64decode(str(sound1)))
        with open("sounds/sound2.mp3", "wb") as fh:
            fh.write(base64.b64decode(str(sound2)))

        mp3_file_1 = AudioSegment.from_file(file="sounds/sound1.mp3", format="mp3")
        mp3_file_2 = AudioSegment.from_file(file="sounds/sound2.mp3", format="mp3")

        play(mp3_file_1)
        play(mp3_file_2)
        return jsonify("OK")
    except Exception as e:
        return jsonify(side=None, error=str(e))


@app.route('/api/merge', methods=['POST'])
def merge_and_play():
    try:
        sound1 = request.get_json(force=True)['sound1']
        sound2 = request.get_json(force=True)['sound2']
        with open("sounds/sound1.mp3", "wb") as fh:
            fh.write(base64.b64decode(str(sound1)))
        with open("sounds/sound2.mp3", "wb") as fh:
            fh.write(base64.b64decode(str(sound2)))

        mp3_file_1 = AudioSegment.from_file(file="sounds/sound1.mp3", format="mp3")
        mp3_file_2 = AudioSegment.from_file(file="sounds/sound2.mp3", format="mp3")

        overlay = mp3_file_1.overlay(mp3_file_2, position=0)
        overlay.export("sounds/output.mp3", format="mp3")  # export

        play(overlay)
        return jsonify("OK")
    except Exception as e:
        return jsonify(side=None, error=str(e))


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
