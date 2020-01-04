import pandas as pd
pd.set_option('display.max_rows', 25)

import numpy as np
import re
import string
import pickle

import os
from os import listdir
from os.path import isfile, join

import itertools

import tensorflow as tf
from keras.models import Model
from keras.utils import np_utils
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.applications.vgg16 import VGG16
from keras_preprocessing.image import ImageDataGenerator

from glob import glob

import PIL.Image
from PIL import Image

from flask import Flask, render_template, url_for, request, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename

###################
# Load DataFrames #
###################

# All recipes
df = pd.read_csv('../pickles-csvs/recipe-tables/df.csv')

# Recipe tables by a variety of health metrics

df_low_cal = pd.read_csv('../pickles-csvs/recipe-tables/low_cal.csv') # low cal
df_low_carb = pd.read_csv('../pickles-csvs/recipe-tables/low_carb.csv') # low carb
df_low_cholesterol = pd.read_csv('../pickles-csvs/recipe-tables/low_cholesterol.csv') # low cholesterol
df_low_sodium = pd.read_csv('../pickles-csvs/recipe-tables/low_sodium.csv') # low sodium
df_low_fat = pd.read_csv('../pickles-csvs/recipe-tables/low_fat.csv') # low fat
df_high_protein = pd.read_csv('../pickles-csvs/recipe-tables/high_protein.csv') # high protein

# Quick recipes

df_quick = pd.read_csv('../pickles-csvs/recipe-tables/df_quick.csv')

# Quick, healthy recipes

df_low_cal_quick = pd.read_csv('../pickles-csvs/recipe-tables/low_cal_quick.csv')
df_low_carb_quick = pd.read_csv('../pickles-csvs/recipe-tables/low_carb_quick.csv')
df_low_cholesterol_quick = pd.read_csv('../pickles-csvs/recipe-tables/low_cholesterol_quick.csv')
df_low_sodium_quick = pd.read_csv('../pickles-csvs/recipe-tables/low_sodium_quick.csv')
df_low_fat_quick = pd.read_csv('../pickles-csvs/recipe-tables/low_fat_quick.csv')
df_high_protein_quick = pd.read_csv('../pickles-csvs/recipe-tables/high_protein_quick.csv')

#####################################

model = load_model('../model.h5') # load model
graph = tf.get_default_graph()

########################################
# Prepare images for model predictions #
########################################

def img_to_array(image_path, size=(150,150)):
    
    # Load images from directory and then convert to numpy array

    image = PIL.Image.open(image_path).resize(size)
    img_data = np.array(image.getdata(), np.float32).reshape(*size, -1)
    img_data = np.flip(img_data, axis=2)
    return img_data

def prepare_image(image_path):
    im = img_to_array(image_path)
    im = im / 255 # account for image augmentation
    im = np.expand_dims(im, axis=0) 

    return im

train_datagen = ImageDataGenerator(
      rescale=1/255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      fill_mode='nearest',
      validation_split = 0.2)

batch_size = 25
target_size = (150, 150)

train_generator = train_datagen.flow_from_directory(
    '../downloads',
    target_size=target_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # training data

validation_generator = train_datagen.flow_from_directory(
    '../downloads',
    target_size=target_size,
    color_mode='rgb',
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation') # validation data

# Assign class labels in order to feed predictions into recipe table

classes = {v: k for k, v in train_generator.class_indices.items()}

##########################
# Neural Net Predictions #
##########################

def image_prediction(images):
    
    prepared_images = []
    outs = []
    predictions = []
    
    # load images, convert to numpy array
    for image in images:
        prepared_images.append(prepare_image('../flask/image_uploads/{}'.format(image)))
    
    # make prediction
    for prep_image in prepared_images:
        outs.append(model.predict(prep_image))
    
    # label prediction
    for out in outs:
        predictions.append(classes[np.argmax(out)])
     
    predictions = [prediction.lower() for prediction in predictions]
    
    return predictions


###########################
# Link model with recipes #
###########################

def linkModelWithRecipes(recipe_table, *predictions):
    
    '''Loop through the model's predictions (as many images as the user uploads) and
    return a table of up to 4 recipes that incorporate those ingredients.
    '''

    try:
        return recipe_table[np.logical_and.reduce([recipe_table['Ingredients'].str.contains(ingredient) for ingredient in predictions])].sample(4)
    except:
        try:
            return recipe_table[np.logical_and.reduce([recipe_table['Ingredients'].str.contains(ingredient) for ingredient in predictions])].sample(3)
        except:
            try:
                return recipe_table[np.logical_and.reduce([recipe_table['Ingredients'].str.contains(ingredient) for ingredient in predictions])].sample(2)
            except:
                try:
                    return recipe_table[np.logical_and.reduce([recipe_table['Ingredients'].str.contains(ingredient) for ingredient in   predictions])].sample(1)
                except:
                    pass
    

########################################
##########FLASK SPECIFIC CODE###########
########################################


UPLOAD_FOLDER = "/Users/samrudovsky/Desktop/Metis/Projects/Neural-Net/flask/image_uploads"


app = Flask(__name__)


app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


@app.route('/')
def upload_form():
    return render_template('Index.html')


@app.route('/recipes', methods=['POST'])
def upload_file():
    
    if request.method == 'POST':
        
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        
        if file:
           
            for f in request.files.getlist('file'):
                filename = secure_filename(f.filename)
                f.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                flash('File successfully uploaded')
                
                # define variables to properly render HTML
                diet_preference = request.form['diet_choices'] 
                prep_time_preference = request.form['prep_time_choice']
            
            global graph
            with graph.as_default():
                images = os.listdir('Image_Uploads')[1:] # image file
                predictions = image_prediction(images) # predict class of food image
                print_image_predictions = (' and ').join(predictions) # to print on recipe page
                
                #########################################################
                # Generate recipes based on dietary and time preference #
                #########################################################  
            
                if prep_time_preference == 'No Preference': # All recipes

                    if diet_preference == '':
                        user_specific_recipes = linkModelWithRecipes(df, *predictions) # master recipe table
                    elif diet_preference == 'low carb':
                        user_specific_recipes = linkModelWithRecipes(df_low_carb, *predictions) # low carb
                    elif diet_preference == 'low calorie':
                        user_specific_recipes = linkModelWithRecipes(df_low_cal, *predictions) # low calorie
                    elif diet_preference == 'low fat':
                        user_specific_recipes = linkModelWithRecipes(df_low_fat, *predictions) # low fat
                    elif diet_preference == 'low sodium':
                        user_specific_recipes = linkModelWithRecipes(df_low_sodium, *predictions) # low sodium
                    elif diet_preference == 'low cholesterol':
                        user_specific_recipes = linkModelWithRecipes(df_low_cholesterol, *predictions) # low cholesterol
                    elif diet_preference == 'high protein':
                        user_specific_recipes = linkModelWithRecipes(df_high_protein, *predictions) # high protein


                elif prep_time_preference == 'Under 45 Minutes': # Recipes under 45 minutes

                    if diet_preference == '':
                        user_specific_recipes = linkModelWithRecipes(df_quick, *predictions)
                    elif diet_preference == 'low carb':
                        user_specific_recipes = linkModelWithRecipes(df_low_carb_quick, *predictions)
                    elif diet_preference == 'low calorie':
                        user_specific_recipes = linkModelWithRecipes(df_low_cal_quick, *predictions)  
                    elif diet_preference == 'low fat':
                        user_specific_recipes = linkModelWithRecipes(df_low_fat_quick, *predictions)
                    elif diet_preference == 'low sodium':
                        user_specific_recipes = linkModelWithRecipes(df_low_sodium_quick, *predictions)
                    elif diet_preference == 'low cholesterol':
                        user_specific_recipes = linkModelWithRecipes(df_low_cholesterol_quick, *predictions)
                    elif diet_preference == 'high protein':
                        user_specific_recipes = linkModelWithRecipes(df_high_protein_quick, *predictions)
        

            return render_template('recipes.html', 
                                   diet_preference=diet_preference,
                                   prep_time_preference=prep_time_preference,
                                   predictions=predictions,
                                   print_image_predictions=print_image_predictions,
                                   recipes=user_specific_recipes.to_html(index=False, escape=False))
             
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)
        
       

if __name__ == '__main__':
    app.run(debug=False)
