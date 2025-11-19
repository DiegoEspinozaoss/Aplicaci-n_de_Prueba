from flask import Flask, request

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        nombre = request.form.get("nombre", "").strip()
        return f"""
            <h2>Buenos días, {nombre} ☀️</h2>
            <a href="/">Volver</a>
        """

    return """
        <h1>Aplicación Minimalista en Azure</h1>
        <p>Ingresa tu nombre:</p>
        <form method="POST">
            <input name="nombre" placeholder="Tu nombre" />
            <button type="submit">Enviar</button>
        </form>
    """

# Azure no necesita ejecutar app.run()
if __name__ == "__main__":
    app.run(debug=True)
