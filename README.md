# Home Cookin': Using Computer Vision to Generate Healthy Recipes

## Project 5 – Metis Data Science Bootcamp  (Weeks 9-12)

### Technical Focus
- Image classification with Keras
	- Convolutional neural network with transfer learning
	- Image augmentation
- Cloud computing
	- AWS
		- GPU instance to train neural network 
		- Concurrent web scraping instances
- Data visualization 
	- Flask app
- Web scraping
	- Beautiful Soup and Selenium
---
### Project Goals
- Generate healthy recipes based on the produce, meat, and dairy in one's fridge

- Using computer vision, accurately recognize 15 different foods

- Connect model's predictions with 65,000 recipes from http://www.allrecipes.com and allow a user to specify specific health metrics and prep time preferences

---
### Process
1) Using a Google API, download 300 images from 15 separate classes of foods. Manually inspect each image to ensure it is an accurate depiction of the food.

2) Build the top layer of a convolutional neural network with a VGG16 base.  Use Keras' ImageDataGenerator to augment images and train the fully connected block.  Train a CNN model on a GPU instance in AWS. 

3) Scrape over ~67,000 recipes from http://www.allrecipes.com on multiple AWS instances.  Acquire information including recipe name, ingredients, prep time, rating, and nutritional information.

4) Link image predictions with recipes that incorporate those foods. Allow the user to specify personalized health metrics including low carb, low fat, low sodium, low calorie, low cholesterol, and high protein recipes.  Provide an option to only see recipes that can be completed in under 45 minutes for weeknight efficiency.

5) Build a Flask App to showcase how the model works.

---
### Results

The CNN model made accurate predictions on the training set 96% of the time. It made accurate predictions on unseen images 83% of the time.  The model made predictions with >90% accuracy, on many unseen food categories, including kale (98%), eggplant (96%), sausage (98%), and and bell peppers (94%), but struggled with other food images such as avocados (74%) and apples (78%). 

The biggest hindrance in model performance was likely the sparse training set.  The model was prone to overfitting. With only 300 images downloaded from Google Images for each food group, often close to 100 were deleted because they were not accurate depictions. As such, the model tended to learn patterns that at times did not generalize to the food group as a whole. It also struggled with foods that were displayed in different ways – for example, an avocado was captured both in its whole form as well as sliced in half. In this scenario, without sufficient training data, the model struggled to make robust predictions. 



### Future Work

As currently built, the app may actually be less efficient than having a user perform a simple Google search for the foods in his/her fridge and then explore the distinct health categories on the recipe webpage.  However, with the advent of smart fridges, an image &#8594; recipe generator app could hold tremendous value. 

Perhaps you are at work and you want to check what ingredients you have in the fridge because tonight is your night to cook dinner for the fam.  Using similar image recognition technology to the CNN employed in this model, the app would inform you of the contents of your fridge.  Next, the recipe generator would search the table of 65,000 recipes and supply you with highly rated recipes that fit your personalized dietary metrics and prep-time constraints. 

Perhaps you have all the ingredients for shakshuka except for the Israeli spices! I could integrate a delivery function into that app that connects with a grocery delivery service such as Instacart. As you finish up your lunch-time chipotle burrito at your cubicle, you can order any last minute provisions for a quick, healthy, and delicious weeknight meal!

