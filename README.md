# Image to Recipe Generator

This project is a web application that allows users to upload images of food items and receive a quick, healthy recipe based on the ingredients detected in the image. The application leverages state-of-the-art machine learning models for image captioning and natural language processing to generate recipes.

## Features

- **Image Captioning**: Utilizes a pre-trained model from Hugging Face to convert images into text descriptions of the ingredients.
- **Recipe Generation**: Employs a language model to generate quick and healthy recipes based on the identified ingredients.
- **Interactive Web Interface**: Built with Streamlit, providing an easy-to-use interface for uploading images and viewing recipes.
- **Nutritional Information**: Calculates and displays the macronutrient breakdown and total calories for each recipe.
- **Customizable**: Users can modify the language model settings and recipe generation parameters to suit their needs.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- An OpenAI API key
- A Hugging Face API token

### Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/image-to-recipe.git
   cd image-to-recipe
   ```

2. **Install Dependencies**

   Use pip to install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables**

   Create a `.env` file in the root directory and add your API keys:

   ```
   OPENAI_API_KEY=your_openai_api_key
   HUGGINFACE_HUB_API_TOKEN=your_huggingface_api_token
   ```

4. **Run the Application**

   Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```

## Usage

1. **Upload an Image**: Use the file uploader to select one or more images of food items.
2. **Generate Recipe**: Click the "GENERATE" button to process the images and generate recipes.
3. **View Results**: The application will display the detected ingredients and the generated recipe, including preparation instructions and nutritional information.

## How It Works

- **Image Processing**: The `image_to_text` function uses the `Salesforce/blip-image-captioning-large` model to extract text descriptions from images.
- **Recipe Generation**: The `generate_recipe` function uses a language model to create a recipe based on the extracted ingredients. It formats the output to include preparation time, difficulty, ingredients, kitchen tools needed, instructions, and macronutrient information.
- **Streamlit Interface**: The `main` function sets up the Streamlit interface, allowing users to interact with the application seamlessly.

