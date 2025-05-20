from django.shortcuts import render

# Create your views here.
import numpy as np
import tensorflow as tf
from django.shortcuts import render


# Load the saved LSTM model once
model = tf.keras.models.load_model('lstm_model.keras')


def predict_view(request):
    prediction = None


    if request.method == 'POST':
        # Example: accept comma-separated float sequence
        input_sequence = request.POST.get('sequence')
        try:
            sequence = [float(x) for x in input_sequence.split(',')]
            sequence = np.array(sequence).reshape((1, len(sequence), 1))  # Reshape for LSTM
            prediction = model.predict(sequence)[0][0]
        except Exception as e:
            prediction = f"Error: {str(e)}"


    return render(request, 'predictor/predict.html', {'prediction': prediction})