from flask import Flask, request, render_template_string
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML_FORM = """
<h1>Subir PDF en Azure</h1>

<form method="POST" enctype="multipart/form-data">
    <label>Sube un PDF:</label><br><br>
    <input type="file" name="pdf_file" accept="application/pdf" required>
    <br><br>
    <button type="submit">Subir</button>
</form>

{% if mensaje %}
    <h3>{{ mensaje }}</h3>
{% endif %}
"""

@app.route("/", methods=["GET", "POST"])
def index():
    mensaje = None

    if request.method == "POST":
        if "pdf_file" not in request.files:
            mensaje = "No se subió ningún archivo."
        else:
            file = request.files["pdf_file"]

            if file.filename == "":
                mensaje = "No seleccionaste un archivo."
            elif not file.filename.lower().endswith(".pdf"):
                mensaje = "Solo se permiten archivos PDF."
            else:
                # Guardar el PDF en la carpeta uploads/
                save_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(save_path)
                mensaje = f"PDF recibido correctamente: {file.filename}"

    return render_template_string(HTML_FORM, mensaje=mensaje)


if __name__ == "__main__":
    app.run(debug=True)
