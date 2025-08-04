Project Overview

app.py is a Python application that uses Large Language Models (LLMs) to:
Generate realistic synthetic real estate listings (at least 10 properties).
Create vector embeddings for listings and store them in a SQLite database.
Perform semantic search to match buyer preferences with listings.
Personalize listing descriptions using LLMs to highlight features that align with buyer preferences, without changing factual details.

This project is Vocareum-compatible and uses a Vocareum-issued OpenAI API key.
Key Features:

Synthetic Data Generation:
Uses gpt-4o-mini to generate diverse property data including neighborhood, price, bedrooms, bathrooms, house size, and descriptions.
Vector Database Storage:
Stores embeddings in listings.db for efficient semantic search.
Semantic Search:
Matches buyer queries (e.g., “4-bedroom house with garden near good schools”) to the most relevant listings.
Personalized Descriptions:
Augments descriptions to highlight features important to the buyer (e.g., proximity to parks, sea view).
Error Handling:
Handles missing files, malformed JSON, and API quota limits gracefully.
File Structure
HomeMatch.py
Main application file containing all functionality:
Generate listings
Create embeddings
Store and search database
Personalize descriptions
Listings.txt
JSON file containing generated listings (or manually created if API quota is exceeded).
listings.db
SQLite database storing embeddings for semantic search.
.env
Environment file storing your Vocareum API key:
OPENAI_KEY=voc-xxxxxxxxxxxxxxxxxxxx
Setup Instructions
1. Install Dependencies
Run:
pip install openai python-dotenv numpy
2. Create .env File
Create a file named .env in the project folder:
OPENAI_KEY=voc-xxxxxxxxxxxxxxxxxxxx
(Replace with your Vocareum API key.)
3. Run the Application
python HomeMatch.py
Usage
The program will generate listings (if Listings.txt does not exist) or load them.
It will create embeddings and store them in listings.db.
When prompted:
Enter your property preferences:
Type your query (examples):
Quiet 4-bedroom house with garden near good schools
Beachfront villa with a pool and sea view
Affordable apartment close to metro station
The program will return personalized matches:
--- Personalized Matches ---

Al Rawdah - $1,200,000
This spacious 4-bedroom home is perfect for families seeking peace and access to top schools...
--------------------------------------------------------------------------------
Manual Testing (if API Quota Exhausted)
If you cannot generate listings due to API limits:
Manually create a Listings.txt file with 10 entries in this format:
[
  {
    "neighborhood": "Al Rawdah",
    "price": 1200000,
    "bedrooms": 3,
    "bathrooms": 2,
    "house_size_sqft": 1800,
    "description": "Modern apartment with sea view, near Corniche.",
    "neighborhood_description": "Upscale area with cafes and schools."
  }
]
