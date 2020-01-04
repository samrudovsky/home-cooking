# Imports

import pandas as pd
import numpy as np
import re
import string
import pickle

import matplotlib.pyplot as plt

import os
from os import listdir
from os.path import isfile, join

import itertools

from sklearn.metrics import confusion_matrix

from keras.models import Model, load_model, Sequential
from keras.utils import np_utils
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.applications.vgg16 import VGG16
from keras_preprocessing.image import ImageDataGenerator
from keras import regularizers

from glob import glob

import PIL.Image
from PIL import Image

# Load data from prior web scraping

recipes_part_1 = pd.read_csv('../pickles-csvs/web-scraping-pickles-csvs/tables_1-6.csv')
recipes_part_2 = pd.read_csv('../pickles-csvs/web-scraping-pickles-csvs/tables_7-12.csv')


# Preprocessing

recipes_part_1.dropna(inplace=True) # Drop recipes without nutrition info
recipes_part_1.drop(columns='Unnamed: 0', inplace=True) # Drop duplicate index
recipes_part_1['Nutrition Info'] = recipes_part_1['Nutrition Info'].apply(lambda x: x[1:-1].split(',')) 

recipes_part_2.drop(columns='Unnamed: 0', inplace=True) # Drop duplicate index
recipes_part_2.dropna(inplace=True) # Drop recipes without nutrition info
recipes_part_2['Nutrition Info'] = recipes_part_2['Nutrition Info'].apply(lambda x: x[1:-1].split(',')) 
recipes_part_2.drop(columns='Unnamed: 0.1', inplace=True) # Drop duplicate index

all_recipes = pd.concat([recipes_part_1, recipes_part_2]) # make table of all recipes

df = all_recipes[all_recipes['Rating'] > 3.5] # Master DataFrame. Only keep high-rated recipes.


# Make discrete nutrition columns from nutrition info

df.loc[:, 'Calories'] = df['Nutrition Info'].map(lambda x: x[0]).astype(float)
df.loc[:, 'Fat'] = df['Nutrition Info'].map(lambda x: x[1]).astype(float)
df.loc[:, 'Carbs'] = df['Nutrition Info'].map(lambda x: x[2]).astype(float)
df.loc[:, 'Protein'] = df['Nutrition Info'].map(lambda x: x[3]).astype(float)
df.loc[:, 'Cholesterol'] = df['Nutrition Info'].map(lambda x: x[4]).astype(float)
df.loc[:, 'Sodium'] = df['Nutrition Info'].map(lambda x: x[5]).astype(float)

df.drop(columns='Nutrition Info', inplace=True) # Drop original column

# Recipe tables by a variety of health metrics

df_low_cal = df[df['Calories'] <= 500] # low cal
df_low_fat = df[df['Fat'] <= 7] # low fat
df_low_carb = df[df['Carbs'] <= 25] # low carb
df_high_protein = df[df['Protein'] >= 20] # high protein
df_low_cholesterol = df[df['Cholesterol'] <= 25] # low cholesterol
df_low_sodium = df[df['Sodium'] <= 500] # low sodium


# Quick Recipes

df_quick = df[df['Prep Time'] <= 45]

# Quick, healthy recipes

df_low_cal_quick = df_quick[df_quick['Calories'] <= 500]
df_low_fat_quick = df_quick[df_quick['Fat'] <= 7] 
df_low_carb_quick = df_quick[df_quick['Carbs'] <= 25] 
df_high_protein_quick = df_quick[df_quick['Protein'] >= 20] 
df_low_cholesterol_quick = df_quick[df_quick['Cholesterol'] <= 25] 
df_low_sodium_quick = df_quick[df_quick['Sodium'] <= 500] 

###############################################

model = load_model('../model.h5') # load model

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


# Augment Image Data
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

classes = {v: k for k, v in train_generator.class_indices.items()} # Assign intelligible class labels

##########################
# Neural Net Predictions #
##########################

images = []

for subdir, dirs, files in os.walk('Image_Uploads'):
    for file in files:
        filepath = subdir + os.sep + file
        if filepath.endswith(".jpg"):
            images.append(filepath.split('/')[1])

images = os.listdir('../flask/image_uploads')[1:]


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
       
    return predictions


predictions = image_prediction(images)
predictions = [prediction.lower() for prediction in predictions]


############################################
# Select dietary and prep time preferences #
############################################

def pickDietaryPreferences():
   
    # dietary preference

    response = input('Do you have any dietary preferences? (yes/no) ')
     
    if response == 'yes':
        print('\nPlease select one of the following: ')
        print('\nLow Carb –– Low Fat –– Low Calorie –– Low Sodium –– Low Cholesterol –– High Protein\n')
        
        dietary_choice = input().lower()
        return dietary_choice
    
    elif response == 'no':
        dietary_choice = 'no'
        return dietary_choice

dietary_choice = pickDietaryPreferences()


def pickPrepTime():
    
    # prep time preference

    response = input('Do you want recipes that can be finished in under 45 minutes? (yes/no) ')
    
    if response == 'yes':
        preptime_choice = 'yes'
        return preptime_choice
    
    return('no')

preptime_choice = pickPrepTime()

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


#########################################################
# Generate recipes based on dietary and time preference #
#########################################################                

if preptime_choice == 'no': # All recipes
    
    if dietary_choice == 'no':
        user_specific_recipes = linkModelWithRecipes(df, *predictions) # master recipe table
    elif dietary_choice == 'low carb':
        user_specific_recipes = linkModelWithRecipes(df_low_carb, *predictions) # low carb
    elif dietary_choice == 'low calorie':
        user_specific_recipes = linkModelWithRecipes(df_low_cal, *predictions) # low calorie
    elif dietary_choice == 'low fat':
        user_specific_recipes = linkModelWithRecipes(df_low_fat, *predictions) # low fat
    elif dietary_choice == 'low sodium':
        user_specific_recipes = linkModelWithRecipes(df_low_sodium, *predictions) # low sodium
    elif dietary_choice == 'low cholesterol':
        user_specific_recipes = linkModelWithRecipes(df_low_cholesterol, *predictions) # low cholesterol
    elif dietary_choice == 'high protein':
        user_specific_recipes = linkModelWithRecipes(df_high_protein, *predictions) # high protein
        

else: # Recipes under 45 minutes

    if dietary_choice == 'no':
        user_specific_recipes = linkModelWithRecipes(df_quick, *predictions)
    elif dietary_choice == 'low carb':
        user_specific_recipes = linkModelWithRecipes(df_low_carb_quick, *predictions)
    elif dietary_choice == 'low calorie':
        user_specific_recipes = linkModelWithRecipes(df_low_cal_quick, *predictions)  
    elif dietary_choice == 'low fat':
        user_specific_recipes = linkModelWithRecipes(df_low_fat_quick, *predictions)
    elif dietary_choice == 'low sodium':
        user_specific_recipes = linkModelWithRecipes(df_low_sodium_quick, *predictions)
    elif dietary_choice == 'low cholesterol':
        user_specific_recipes = linkModelWithRecipes(df_low_cholesterol_quick, *predictions)
    elif dietary_choice == 'high protein':
        user_specific_recipes = linkModelWithRecipes(df_high_protein_quick, *predictions)

print('\nTry these recipes!\n')
print(user_specific_recipes)

