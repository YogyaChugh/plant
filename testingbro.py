from flask import Flask, render_template, request, jsonify
import io
import test_model

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html',prediction="")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"})
    file = request.files.get('file')
    if file.filename == '':
        return jsonify({"error": "Empty filename"})
    try:
        file.seek(0)
        image_bytes = file.read()
        pillow_img = io.BytesIO(image_bytes)
        prediction = test_model.aibro(pillow_img)
        if prediction:
            print(prediction)
        else:
            print("not returning prediction ")
        return jsonify({"prediction":prediction.replace("_"," ")})
    except Exception as e:
        print("error: ",str(e))
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
