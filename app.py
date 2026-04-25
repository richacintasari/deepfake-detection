from flask import Flask, render_template, request, session
import os, random

app = Flask(__name__)
app.secret_key = "secret123"

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png','jpg','jpeg','mp4'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(filepath):
    # simulasi AI + breakdown
    score = random.uniform(70, 99)
    result = "REAL" if random.random() > 0.5 else "FAKE"

    breakdown = {
        "Visual Analysis": round(random.uniform(70, 95), 2),
        "Texture Consistency": round(random.uniform(60, 90), 2),
        "Lighting Analysis": round(random.uniform(70, 95), 2),
    }

    reason = "Kemungkinan manipulasi pada area wajah / tekstur tidak konsisten" if result=="FAKE" \
             else "Tidak ditemukan indikasi manipulasi signifikan"

    return result, round(score,2), breakdown, reason

@app.route('/', methods=['GET','POST'])
def index():
    result = confidence = filename = None
    breakdown = reason = None
    error = None

    if "history" not in session:
        session["history"] = []

    if request.method == 'POST':
        file = request.files.get('file')

        if not file or file.filename == '':
            error = "File tidak ditemukan"
        elif not allowed_file(file.filename):
            error = "Format file tidak didukung (png, jpg, jpeg, mp4)"
        else:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            result, confidence, breakdown, reason = predict_image(filepath)
            filename = file.filename

            # simpan history (max 5)
            session["history"] = ([{
                "name": filename,
                "result": result,
                "confidence": confidence
            }] + session["history"])[:5]

    return render_template(
        'index.html',
        result=result,
        confidence=confidence,
        filename=filename,
        breakdown=breakdown,
        reason=reason,
        history=session.get("history", []),
        error=error
    )

if __name__ == '__main__':
    app.run(debug=True)