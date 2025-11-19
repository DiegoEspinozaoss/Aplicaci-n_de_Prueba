import os
from flask import Flask, request, render_template
from rag_runner import run_rag   # 🔥 Tu pipeline RAG real

# Crear carpeta para subir PDFs
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    # Vista con formulario para subir PDF
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

    # Ejecutar tu RAG 🚀
    resultados = run_rag(pdf_path)

    # Mostrar los resultados en HTML
    return render_template("resultado.html", resultados=resultados)


if __name__ == "__main__":
    app.run(debug=True)
