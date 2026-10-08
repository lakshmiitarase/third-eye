from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    send_from_directory
)

import sqlite3
import os

from generator import generate_face
from description_parser import extract_features


# =========================================================
# APPLICATION SETUP
# =========================================================

print("Current Working Directory:", os.getcwd())

app = Flask(__name__)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# LOGIN
# =========================================================

@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        investigator_id = request.form.get(
            "investigator_id",
            ""
        ).strip()

        password = request.form.get(
            "password",
            ""
        )

        conn = sqlite3.connect("third_eye.db")
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT *
            FROM investigator
            WHERE investigator_id = ?
            AND password = ?
            """,
            (investigator_id, password)
        )

        user = cursor.fetchone()

        conn.close()

        if user:
            return redirect(url_for("witness"))

        return "Invalid Investigator ID or Password"

    return render_template("login.html")


# =========================================================
# REGISTER
# =========================================================

@app.route("/register")
def register():
    return render_template("register.html")


# =========================================================
# CREATE ACCOUNT
# =========================================================

@app.route("/createaccount")
def createaccount():
    return render_template("createaccount.html")


# =========================================================
# SAVE REGISTRATION
# =========================================================

@app.route("/save_registration", methods=["POST"])
def save_registration():

    investigator_id = request.form.get(
        "investigator_id",
        ""
    ).strip()

    agency = request.form.get(
        "agency",
        ""
    ).strip()

    password = request.form.get(
        "password",
        ""
    )

    if not investigator_id or not agency or not password:
        return "All registration fields are required."

    conn = sqlite3.connect("third_eye.db")
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO investigator
            (investigator_id, agency, password)
            VALUES (?, ?, ?)
            """,
            (
                investigator_id,
                agency,
                password
            )
        )

        conn.commit()

    except sqlite3.IntegrityError:

        conn.close()

        return "Investigator ID already exists."

    conn.close()

    print("Registration Successful")

    return redirect(url_for("login"))


# =========================================================
# WITNESS
# =========================================================

@app.route("/witness")
def witness():
    return render_template("witness.html")


# =========================================================
# SAVE WITNESS
# =========================================================

@app.route("/save_witness", methods=["POST"])
def save_witness():

    print("Step A: save_witness called")

    witness_name = request.form.get(
        "witness_name",
        ""
    ).strip()

    description = request.form.get(
        "description",
        ""
    ).strip()

    if not description:
        return "Suspect description is required."

    conn = sqlite3.connect("third_eye.db")
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO witness
        (witness_name, description)
        VALUES (?, ?)
        """,
        (
            witness_name,
            description
        )
    )

    conn.commit()
    conn.close()

    print("Step B: Data saved to database")

    return redirect(url_for("generate"))


# =========================================================
# GENERATE FACE
# =========================================================

@app.route("/generate", methods=["GET", "POST"])
def generate():

    print("Step C: generate route called")

    # -----------------------------------------------------
    # GET DESCRIPTION FROM FORM
    # -----------------------------------------------------

    if request.method == "POST":

        description = request.form.get(
            "description",
            ""
        ).strip()

    # -----------------------------------------------------
    # GET LATEST DESCRIPTION FROM DATABASE
    # -----------------------------------------------------

    else:

        conn = sqlite3.connect("third_eye.db")
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT description
            FROM witness
            ORDER BY id DESC
            LIMIT 1
            """
        )

        row = cursor.fetchone()

        conn.close()

        if row:
            description = row[0]
        else:
            description = ""

    # -----------------------------------------------------
    # CHECK DESCRIPTION
    # -----------------------------------------------------

    if not description:

        return "No suspect description was provided."

    print("\nDescription received:")
    print(description)

    # =====================================================
    # GENERATE IMAGE
    # =====================================================

    try:

        # Convert description into feature attributes
        attributes = extract_features(description)
 
        print("\nExtracted attributes:")
        print(attributes)

        # Generate face using the extracted attributes
        print("ATTRIBUTES SENT TO GENERATOR:")
        print(attributes)
        
        image_path = generate_face(
                attributes
        )

        print("\nGenerated image path:")
        print(image_path)

    except Exception as e:

        import traceback

        print("\nERROR DURING FACE GENERATION:")
        print(e)
        traceback.print_exc()

        return (
             "<h3>Face generation failed</h3>"
             f"<p>{e}</p>"
        )

    # =====================================================
    # SHOW RESULT
    # =====================================================

    return render_template(
        "result.html",
        image=image_path
    )


# =========================================================
# RESULT
# =========================================================

@app.route("/result")
def result():
    return render_template("result.html")


# =========================================================
# SERVE GENERATED IMAGES
# =========================================================

@app.route("/outputs/<path:filename>")
def generated_image(filename):

    output_folder = os.path.join(
        app.root_path,
        "static",
        "outputs"
    )

    return send_from_directory(
        output_folder,
        filename
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False
    )