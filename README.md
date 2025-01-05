# Opinno
Work I did for Opinno Ecuador I ommitted all the data I was using which is sensitive and this is just the logic and codes that I implemented 

This is separated into three main projects that I worked on. The matching project which is a Jupyter notebook, 
a categorization project in docker which is docker-py-cat and a natural language processing project.

1. The matching problem was that we had products from a company and a lot of receipts from different distributors and the purpose was to find the products sold but there were problems

   - the first problem was that the names of the products did not match because distributors used different names
   - the prices did not match because distributors used different discounts but did not mention them or sell the products in different manners, for example if you have a pack that contains 5 of a          product they would split it open and sell each one individually even if they bought it in one form.
   - The product id was different from the suppliers and distributors

In order to solve this problem my approach was to find a similarity between the id, the name and the price sold of each product and give these parameters weights in order to find the closest match

the name matching was divided into two parts specific words and the whole name. I used fuzzy matching and lemmatizing in order to find the closest match

for the price match is was a simple number comparison to see which prices matched it the most.

for the id it would try to find the id of one in the other because there was cases in which the id from the distributor was a longer version of the id from the supplier. for example 12345 and 123

2. The categorization problem was that I needed to categorize products sold into specific categories given by the government for tax writeoff purposes and for the data to be better organized.

   - the problem was that the categories were broad and unspecific because there are things which are difficult to categorize. For example you have a category which is health and a category which is       household items and consumables, does toothpaste count aws a household item or as a health item. Where do you put gasoline or very specific and niche objects.

In order to tackle this problem I first tried to get clean data to train a machine learning algorithm which would categorize each product as it got more data but again I ran into problems. 
I did not have clean data that I could use, in spanish since the products were in spanish and it was difficult to find catergorized data if that is what Im trying to do.

In order to fix this I grabbed data from places I know sell things from a specific category a hospital will probably not sell clothes kind of thing. 

This kind of worked I still needed to clean it a lot by hand because sometimes a pharmacy also sells candy for whatever reason then I used the data to train different types of models and verified which one was the best.

3. The third project required me to find correlations between words or sentences to categorize things sold again in the same categories but now things sold can be things like trips or full on events

for this I used natural language processing and levenshteins distance to make natural language processing models in spanish. This was a little more straightforward but the results were not perfect I got an accuracy of around 87 percent.

