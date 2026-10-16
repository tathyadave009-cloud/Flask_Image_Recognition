"""Integration tests for happy path scenarios."""
from io import BytesIO
from PIL import Image


def test_home_page(client):
    """Test that the home page (GET /) loads successfully with status code 200."""
    response = client.get("/")
    assert response.status_code == 200
    assert b"Hand Sign Digit Recognition" in response.data or b"Upload" in response.data


def test_successful_prediction_mocked(client, mocker):
    """Test the successful image upload and prediction with mocked model prediction."""
    mocker.patch("app.predict_result", return_value=5)

    img = Image.new("RGB", (100, 100), color="white")
    img_byte_arr = BytesIO()
    img.save(img_byte_arr, format="JPEG")
    img_byte_arr.seek(0)

    response = client.post(
        "/prediction",
        data={"file": (img_byte_arr, "sample_5.jpeg")},
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"5" in response.data


def test_successful_prediction(client):
    """Test the successful image upload and prediction route response."""
    img_data = BytesIO(b"fake_image_data")
    img_data.name = "test.jpg"

    response = client.post(
        "/prediction",
        data={"file": (img_data, img_data.name)},
        content_type="multipart/form-data"
    )

    assert response.status_code == 200
    assert b"Prediction" in response.data
