"""
Flask Image Recognition Application.
Provides web endpoints for uploading digit images and serving ML predictions.
"""
from flask import Flask, render_template, request
from model import preprocess_img, predict_result

# Instantiating flask app
app = Flask(__name__)


# Home route
@app.route("/")
def main():
    """Renders the main landing page with file upload form."""
    return render_template("index.html")


# Prediction route
@app.route('/prediction', methods=['POST'])
def predict_image_file():
    """
    Handles file upload, invokes preprocessing and prediction,
    and returns the rendered prediction result.
    """
    if "file" not in request.files or not request.files["file"].filename:
        return render_template("result.html", err="File cannot be processed.")

    try:
        img = preprocess_img(request.files['file'].stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))
    except Exception:  # pylint: disable=broad-exception-caught
        error = "File cannot be processed."
        return render_template("result.html", err=error)


# Driver code
if __name__ == "__main__":
    app.run(port=9000, debug=True)
