# Google API to download Google Images to local hard disk

from google_images_download import google_images_download

# Lists of foods to download

veggies = ['Broccoli', 'Spinach', 'Kale', 'Bell Pepper', 'Asparagus',
            'Sweet Potato', 'Cucumber']

fruits = ['Avocado', 'Apple fruit', 'Banana', 'Tomato', 'Lemon', 'Blueberries', 'Strawberries', 'Grapes']

meat_and_fish = ['Ground Beef', 'Steak', 'Pork Chop', 'Bacon', 'Sausage', 
                 'Chicken Breast', 'Salmon', 'Chicken Eggs']

starches = ['Sliced Bread', 'Rice', 'Pasta', 'Oatmeal']


def downloadFoodImages(list_foods, n_img=300):
    '''
    Use Google API to download 300 images for each class and save to disk. 
    Make each food the title of a subdirectory in the 
    'downloads' subdirectory for image augmentation and model training
    '''
    for food in list_foods:
        arguments = {"keywords": food,
                     "limit": n_img,
                     "print_urls": False,
                     "chromedriver": "/Applications/chromedriver"} 
        response = google_images_download.googleimagesdownload()  
        paths = response.download(arguments)


downloadFoodImages(fruits) # 300 images for each fruit
downloadFoodImages(veggies) # 300 images for each vegetable
downloadFoodImages(meat_and_fish) # 300 images for each meat/fish
downloadFoodImages(starches) # 300 images for each starch 

