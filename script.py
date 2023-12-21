from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load Model h5 (path ke model)
genre_model = load_model('genre_model.h5')
genre_labels = { # labels
    0: 'abstract',
    1: 'design',
    2: 'figurative',
    3: 'illustration',
    4: 'landscape',
    5: 'nude painting (nu)',
    6: 'portrait',
    7: 'religious painting',
    8: 'sketch and study',
    9: 'symbolic painting'
}

style_model = load_model('style_model.h5')
style_labels = {
    0: 'Rococo',
    1: 'HighRenaissance',
    2: 'Shin-hanga',
    3: 'NorthernRenaissance', 
    4: 'MagicRealism',
    5: 'Symbolism',
    6: 'Ukiyo-e',
    7: 'Photorealism',
    8: 'FantasticRealism',
    9: 'cartoon',
    10: 'Neo-baroque',
    11: 'Impressionism',
    12: 'FeministArt',
    13: 'Cubo-Futurism',
    14: 'Constructivism',
    15: 'PopArt',
    16: 'Naturalism',
    17: 'NewEuropeanPainting',
    18: 'Divisionism',
    19: 'Academicism',
    20: 'Cubism',
    21: 'Suprematism',
    22: 'Tonalism',
    23: 'ArtNouveau(Modern)',
    24: 'photo',
    25: 'ArtDeco',
    26: 'Realism'
}

era_model = load_model('era_model.h5')
era_labels = {
    0: 'baroque', 
    1: 'contemporary', 
    2: 'impressionism', 
    3: 'medieval', 
    4: 'modern', 
    5: 'neoclassicism', 
    6: 'post impressionism', 
    7: 'primitivism', 
    8: 'realism', 
    9: 'renaissance', 
    10: 'rococo', 
    11: 'romanticism'
}



# preprocess image
def prepare_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Ambil class dengan probability tertinggi dan map index ke corresponding label
def get_highest_probability_class(predictions, class_labels): 
    class_index = np.argmax(predictions)
    class_label = class_labels.get(class_index, 'Unknown')
    return class_index

# endpoint HTTP Post request untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():

    data = {"success": False, "predictions": {}} # Initialize dictionary sebagai response 

    # cek apakah ada file "image" di request
    if "image" in request.files:
        image_file = request.files["image"]
        if image_file.filename == "":
            return jsonify(data)

        # save image yg sudah diupload ke folder baru 
        image_path = f"uploads/{image_file.filename}"
        image_file.save(image_path)

        # preproses imagenya  
        processed_image = prepare_image(image_path)

        # Prediksi model genre
        predictions_genre = genre_model.predict(processed_image)
        predicted_genre = get_highest_probability_class(predictions_genre, genre_labels)
        data["predictions"]["genre_model"] = predicted_genre # Update dictionary dengan hasil prediction

        # Prediksi model style
        predictions_style = style_model.predict(processed_image)
        predicted_style = get_highest_probability_class(predictions_style, style_labels)
        data["predictions"]["style_model"] = predicted_style # Update dictionary dengan hasil prediction

        # Prediksi model era
        predictions_era = era_model.predict(processed_image)
        predicted_era = get_highest_probability_class(predictions_era, era_labels)
        data["predictions"]["era_model"] = predicted_era # Update dictionary dengan hasil prediction
        
        
        data["success"] = True

        os.remove(image_path)

    return jsonify(data)


if __name__ == '__main__':
    app.run(port=5000)
