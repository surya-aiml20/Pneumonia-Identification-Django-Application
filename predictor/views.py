import tensorflow as tf
from django.shortcuts import render
from django.http import JsonResponse
from PIL import Image
import numpy as np
from django.views.decorators.csrf import csrf_exempt

# Load the deep learning model
MODEL_PATH = 'predictor/model/best_pneumonia_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)


def index(request):
    """Render the homepage."""
    return render(request, 'predictor/index.html')


@csrf_exempt
def predict_pneumonia(request):
    """Handle file upload and make predictions."""
    if request.method == 'POST' and request.FILES['file']:
        # Get the uploaded file
        uploaded_file = request.FILES['file']

        # Open the image, convert to RGB, and resize to (224, 224)
        image = Image.open(uploaded_file).convert('RGB').resize((224, 224))
        image_array = np.expand_dims(np.array(image) / 255.0, axis=0)  # Normalize and add batch dimension

        # Make prediction using the model
        prediction = model.predict(image_array)[0]  # Get the first (and only) prediction in the batch
        pneumonia_percentage = float(round(prediction[0] * 100, 2))  # Convert to Python float
        normal_percentage = float(round((1 - prediction[0]) * 100, 2))  # Convert to Python float

        # Determine the result
        if prediction[0] > 0.5:
            result = 'Pneumonia Detected'
            advice = (
                "Please consult a doctor immediately. Follow these steps:\n"
                "- Get a chest X-ray for further confirmation.\n"
                "- Ensure proper medication is started.\n"
                "- Stay hydrated and rest well.\n"
                "- Follow your doctor's advice."
            )
        else:
            result = 'Normal'
            advice = (
                "Your lungs seem healthy. To maintain good health:\n"
                "- Continue a healthy diet and lifestyle.\n"
                "- Avoid smoking and air pollutants.\n"
                "- Keep up with regular exercise and hydration.\n"
                "- Visit a doctor for routine check-ups."
            )

        return JsonResponse({
            'result': result,
            'pneumonia_percentage': pneumonia_percentage,
            'normal_percentage': normal_percentage,
            'advice': advice,
        })

    return JsonResponse({'error': 'Invalid request'}, status=400)