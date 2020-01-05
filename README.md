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
- Generate healthy recipes based on the food in one's fridge

- Using computer vision, accurately recognize 13 different foods

- Connect model's predictions with 67,000 recipes from https://www.allrecipes.com and allow a user to specify personalized health metrics and prep time preferences

---
### Process
1) Using a Google API, download 300 images from 13 separate classes of foods. Manually inspect each image to ensure robustness of training data.

2) Build the top layer of a convolutional neural network with a VGG16 base.  Use Keras' ImageDataGenerator to augment images and train a fully connected block.  Train a CNN model on a GPU instance in AWS. 

3) Scrape over 67,000 recipes from https://www.allrecipes.com on multiple AWS instances.  Capture the recipe name, ingredients, prep time, rating, and nutritional information.

4) Link image predictions with recipes that incorporate those foods. Allow a user to specify personalized health metrics including low carb, low fat, low sodium, low calorie, low cholesterol, and high protein recipes.  Also filter recipes that can be completed in under 45 minutes for weeknight efficiency.

5) Build a Flask App to showcase how the model works.

---
### Results

The CNN model made accurate predictions on the training set 97% of the time and achieved an accuracy of 88% on unseen images.  The model made predictions with >90% accuracy on many unseen food classes, including kale (99%), eggplant (96%), sausage (99%), and bell peppers (96%), but struggled with other food images such as avocados (81%) and asparagus (67%).
The model was prone to overfitting due to the spare training seet.  Of the 300 images for each food item, nearly 100 were deleted in each class because they were not accurate depictions. As such, the model learned patterns that at times did not generalize to the class as a whole. It also struggled with foods that were displayed in different ways – i.e. a halved avocado is quite distinct from a whole avocado.  

---

![](app-demo.mov)

### Future Work

With the advent of smart fridge technology, an image &#8594; recipe generator app could hold tremendous value. 

Perhaps tonight is your night to cook dinner for the fam. You're at work and you want to check what ingredients you have in the fridge. Using similar image recognition technology to the CNN employed in this model, the app could inform you of the contents of your fridge display a selection of the thousands of highly rated recipes that suit your individualized health metrics.

Perhaps you have all the ingredients for shakshuka except for the Israeli spices! I could integrate a delivery function into the app that connects with a grocery delivery service such as Instacart. As you finish up your lunch-time chipotle burrito at your cubicle, you can order any last minute provisions for a quick, healthy, and delicious weeknight meal!

