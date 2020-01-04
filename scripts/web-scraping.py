import requests
import time
from bs4 import BeautifulSoup
from recipe_scrapers import scrape_me 
import pickle

with open("links.pkl", 'rb') as f: # pickle file containing 67,000 distinct recipe URLs 
    list_of_links = pickle.load(f)

####################
# Nutritional Info #
####################

def nutritionInfo(link):
    '''
    Scrape distinct nutritional info from each recipe –
    calories, fat, carbs, cholesterol, sodium, and protein.
    '''
    page = requests.get(link).text
    soup = BeautifulSoup(page, 'html5')
    nutrition_facts = soup.find(attrs={'class': 'nutrition-summary-facts'})

    try:
        calories = float(nutrition_facts.find(attrs={'itemprop': 'calories'}).text.split()[0])
        fat = float(nutrition_facts.find(attrs={'itemprop': 'fatContent'}).text.strip())
        carbs = float(nutrition_facts.find(attrs={'itemprop': 'carbohydrateContent'}).text.strip())
        protein = float(nutrition_facts.find(attrs={'itemprop': 'proteinContent'}).text.strip())
        cholesterol = float(nutrition_facts.find(attrs={'itemprop': 'cholesterolContent'}).text.strip())
        sodium = float(nutrition_facts.find(attrs={'itemprop': 'sodiumContent'}).text.strip())
  
        return([calories, fat, carbs, protein, cholesterol, sodium])

    except:
        pass

#################
# Recipe rating #
#################

def recipeRating(link):
    page = requests.get(link).text
    soup = BeautifulSoup(page, 'html5')

    try:
        rating = soup.find(attrs={'itemprop': 'aggregateRating'}).find_all('meta')
        rating_review_split = [i['content'].split() for i in rating]
        rating = float(rating_review_split[0][0])
   
        return rating

    except:
        pass

###########################
# Total number of reviews #
###########################

def numReviews(link):
    page = requests.get(link).text
    soup = BeautifulSoup(page, 'html5')

    try:
        rating = soup.find(attrs={'itemprop': 'aggregateRating'}).find_all('meta')
        rating_review_split = [i['content'].split() for i in rating]
        num_reviews = float(rating_review_split[1][0])

        return num_reviews
    
    except:
        pass

###########################################

# Instantiate empty lists to append recipe information
title = []
ingredients = []
prep_time = []
rating = []
num_reviews = []
nutrition_info = []

c=0

for link in list_of_links: # 67,000 URLs

    # Ryan Lee's scraper
    scraper = scrape_me(link)
    title.append(scraper.title())
    ingredients.append(scraper.ingredients())
    prep_time.append(scraper.total_time())

    # Ratings
    rating.append(recipeRating(link))
    num_reviews.append(numReviews(link))

    # Ingredients
    nutrition_info.append(nutritionInfo(link))

    # Don't get blocked
    if c % 500 == 0:
        time.sleep(300)
        print(c)

    c += 1
