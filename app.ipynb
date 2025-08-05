"""
HomeMatch - Personalized Real Estate Agent
Enhanced version (Vocareum-compatible):
- Generates detailed real estate listings (LLM)
- Stores listings + embeddings in SQLite database
- Performs semantic search based on buyer preferences
- Augments (personalizes) listing descriptions with LLM
- Handles key mismatches, JSON parsing errors, and missing fields
"""

# ------------------ Imports & Setup ------------------
try:
    import os
    import json
    import sqlite3
    import re

    import numpy as np
    from numpy.linalg import norm
    from dotenv import load_dotenv
    from openai import OpenAI, AuthenticationError
except ModuleNotFoundError as e:
    print(f"Missing module: {e.name}. Install it using `pip install {e.name}`.")
    exit(1)

# Load environment variables
load_dotenv()

API_KEY = os.getenv("OPENAI_KEY")
if not API_KEY:
    raise ValueError(
        "OPENAI_KEY not found in .env file. Please add it and rerun."
    )


# ------------------ Create OpenAI Client ------------------
def create_client(api_key):
    """
    Initialize and verify OpenAI client using Vocareum endpoint.
    """
    try:
        client = OpenAI(
            base_url="https://openai.vocareum.com/v1",  # Vocareum routing
            api_key=api_key,
        )
        # Lightweight check (models.list is not supported in Vocareum)
        client.embeddings.create(input="test", model="text-embedding-3-small")
        return client
    except AuthenticationError:
        print("Invalid Vocareum API key. Please check your .env file.")
        return None
    except Exception as e:
        print(f"Error verifying Vocareum API key: {e}")
        return None


client = create_client(API_KEY)


# ------------------ Key Normalization ------------------
def normalize_listing_keys(listing):
    """
    Normalize keys from model output to expected lowercase keys.

    Args:
        listing (dict): Original listing with possible inconsistent keys.

    Returns:
        dict: Listing with standardized keys.
    """
    return {
        "neighborhood": listing.get("Neighborhood") or listing.get("neighborhood", ""),
        "price": listing.get("Price") or listing.get("price", 0),
        "bedrooms": listing.get("Bedrooms") or listing.get("bedrooms", 0),
        "bathrooms": listing.get("Bathrooms") or listing.get("bathrooms", 0),
        "house_size_sqft": listing.get("HouseSize")
        or listing.get("house_size_sqft", 0),
        "description": listing.get("Description") or listing.get("description", ""),
        "neighborhood_description": listing.get("NeighborhoodDescription")
        or listing.get("neighborhood_description", ""),
    }


# ------------------ Extract JSON Helper ------------------
def extract_json_from_text(text):
    """
    Attempt to extract JSON array from text if model output contains extra text.

    Args:
        text (str): The raw string output from the LLM.

    Returns:
        list: Extracted JSON array or an empty list if parsing fails.
    """
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return []
    return []


# ------------------ 1. Generate Listings ------------------
def generate_listings(client, count=10):
    """
    Generate realistic real estate listings using LLM.

    Args:
        client (OpenAI): Authenticated OpenAI client.
        count (int): Number of listings to generate.

    Returns:
        list: Generated listings or empty list on failure.
    """
    prompt = f"""
    Generate {count} diverse and realistic real estate listings in Saudi Arabia.
    Return ONLY a JSON array. Each object MUST have these exact keys:
    - neighborhood (string)
    - price (number)
    - bedrooms (integer)
    - bathrooms (integer)
    - house_size_sqft (integer)
    - description (string)
    - neighborhood_description (string)

    Do not omit any keys, even if values are estimates.
    Do not include any text outside of the JSON array.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a real estate data generator."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.8,
    )

    text = response.choices[0].message.content

    # Attempt direct JSON parse
    try:
        listings = json.loads(text)
    except json.JSONDecodeError:
        listings = extract_json_from_text(text)

    if not listings:
        print("\n[ERROR] Failed to parse JSON from LLM output.")
        print("Raw output received:\n", text)
        print("\nTo continue testing, create 'Listings.txt' manually with sample JSON data.")
        return []

    # Save to file
    with open("Listings.txt", "w", encoding="utf-8") as f:
        json.dump(listings, f, indent=2, ensure_ascii=False)

    print(f"Generated {len(listings)} listings → saved to Listings.txt")
    return listings


# ------------------ 2. Create Embeddings + Store in DB ------------------
def create_embeddings_and_store(client, listings, db_path="listings.db"):
    """
    Create embeddings for listings and store everything in SQLite DB.

    Args:
        client (OpenAI): Authenticated OpenAI client.
        listings (list): List of property listings.
        db_path (str): Path to SQLite database file.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS listings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            neighborhood TEXT,
            price REAL,
            bedrooms INTEGER,
            bathrooms INTEGER,
            house_size_sqft INTEGER,
            description TEXT,
            neighborhood_description TEXT,
            embedding TEXT
        )
    """
    )

    cursor.execute("DELETE FROM listings")

    for listing in listings:
        listing = normalize_listing_keys(listing)

        description = listing.get("description", "")
        neighborhood_desc = listing.get("neighborhood_description", "")

        if not description or not neighborhood_desc:
            print(f"Skipping listing due to missing fields: {listing}")
            continue

        combined = description + " " + neighborhood_desc

        response = client.embeddings.create(
            input=combined,
            model="text-embedding-3-small",
        )
        embedding = response.data[0].embedding

        cursor.execute(
            """
            INSERT INTO listings (
                neighborhood, price, bedrooms, bathrooms, house_size_sqft,
                description, neighborhood_description, embedding
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                listing["neighborhood"],
                listing["price"],
                listing["bedrooms"],
                listing["bathrooms"],
                listing["house_size_sqft"],
                description,
                neighborhood_desc,
                json.dumps(embedding),
            ),
        )

    conn.commit()
    conn.close()
    print(f"Stored {len(listings)} listings with embeddings in {db_path}")


# ------------------ 3. Semantic Search ------------------
def semantic_search(client, query, db_path="listings.db", top_k=3):
    """
    Perform semantic search using query embedding against DB embeddings.

    Args:
        client (OpenAI): Authenticated OpenAI client.
        query (str): Buyer preference text query.
        db_path (str): Path to SQLite database.
        top_k (int): Number of top matches to return.

    Returns:
        list: Top matching listings.
    """
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small",
    )
    query_embedding = np.array(response.data[0].embedding)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT neighborhood, price, bedrooms, bathrooms, house_size_sqft,
               description, neighborhood_description, embedding
        FROM listings
    """
    )
    rows = cursor.fetchall()
    conn.close()

    results = []
    for row in rows:
        embedding = np.array(json.loads(row[7]))
        similarity = np.dot(query_embedding, embedding) / (
            norm(query_embedding) * norm(embedding)
        )
        listing = {
            "neighborhood": row[0],
            "price": row[1],
            "bedrooms": row[2],
            "bathrooms": row[3],
            "house_size_sqft": row[4],
            "description": row[5],
            "neighborhood_description": row[6],
        }
        results.append((similarity, listing))

    results.sort(key=lambda x: x[0], reverse=True)
    return [listing for _, listing in results[:top_k]]


# ------------------ 4. Personalize Descriptions ------------------
def personalize_descriptions(client, buyer_preferences, listings, top_k=3):
    """
    Augment listing descriptions using LLM to highlight buyer preferences.

    Args:
        client (OpenAI): Authenticated OpenAI client.
        buyer_preferences (str): User input describing preferences.
        listings (list): Listings to personalize.
        top_k (int): Number of top listings to personalize.

    Returns:
        list: Listings with personalized descriptions.
    """
    personalized = []

    for listing in listings[:top_k]:
        prompt = f"""
        Buyer preferences: {buyer_preferences}

        Listing details:
        Neighborhood: {listing['neighborhood']}
        Price: {listing['price']}
        Bedrooms: {listing['bedrooms']}
        Bathrooms: {listing['bathrooms']}
        House Size: {listing['house_size_sqft']} sqft
        Description: {listing['description']}

        Task:
        Rewrite the listing description to make it appealing and emphasize aspects 
        that align with the buyer's preferences. Do NOT alter factual details like 
        price, number of rooms, or size. Focus on highlighting relevant features.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a professional real estate copywriter."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
        )

        listing["personalized_description"] = response.choices[0].message.content.strip()
        personalized.append(listing)

    return personalized


# ------------------ Main ------------------
if __name__ == "__main__":
    if not client:
        print(
            "Error: OpenAI client not initialized. "
            "Check your Vocareum API key in .env."
        )
        exit(1)

    # Step 1: Load or Generate Listings
    listings = []  # Initialize to avoid NameError
    if os.path.exists("Listings.txt"):
        try:
            with open("Listings.txt", "r", encoding="utf-8") as f:
                listings = json.load(f)
        except json.JSONDecodeError:
            print("Listings.txt is corrupted. Regenerating listings...")
            listings = generate_listings(client, count=10)
    else:
        listings = generate_listings(client, count=10)

    if not listings:
        print("No listings available. Check API quota or JSON parsing errors.")
        exit(1)

    # Step 2: Create embeddings and store in DB
    try:
        create_embeddings_and_store(client, listings)
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        exit(1)

    # Step 3: Get buyer preferences
    buyer_query = input("\nEnter your property preferences: ").strip()
    if not buyer_query:
        print("No preferences entered. Exiting.")
        exit(1)

    # Step 4: Semantic search
    try:
        matches = semantic_search(client, buyer_query)
    except Exception as e:
        print(f"Error during semantic search: {e}")
        exit(1)

    if not matches:
        print("No matches found for your query.")
        exit(0)

    # Step 5: Personalize descriptions
    try:
        personalized = personalize_descriptions(client, buyer_query, matches)
    except Exception as e:
        print(f"Error personalizing descriptions: {e}")
        personalized = matches

    # Step 6: Display results
    print("\n--- Personalized Matches ---\n")
    for listing in personalized:
        print(f"{listing['neighborhood']} - ${listing['price']}")
        print(listing.get("personalized_description", listing["description"]))
        print("-" * 80)
