import os
from flask import Flask, request, render_template
from rag_runner import run_rag

# Rutas absolutas (Azure necesita esto)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return render_template("upload.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "pdf_file" not in request.files:
        return "No se subió ningún archivo."

    file = request.files["pdf_file"]

    if file.filename == "":
        return "No seleccionaste un archivo."

    if not file.filename.lower().endswith(".pdf"):
        return "Solo se permiten archivos PDF."

    # Guardar PDF
    pdf_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(pdf_path)

    # Ejecutar tu RAG
    resultados = run_rag(pdf_path)

    return render_template("resultado.html", resultados=resultados)


# Gunicorn ignorará este bloque en Azure
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
