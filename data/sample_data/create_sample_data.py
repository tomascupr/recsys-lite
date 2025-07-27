from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Parameters
n_users = 100
n_items = 500
n_events = 10000
n_categories = 10
n_brands = 20

# Create user IDs
user_ids = [f"U_{str(i).zfill(2)}" for i in range(n_users)]

# Create item IDs
item_ids = [f"I_{str(i).zfill(4)}" for i in range(n_items)]

# Create events data
event_user_ids = np.random.choice(user_ids, size=n_events)
event_item_ids = np.random.choice(item_ids, size=n_events)
end_date = datetime.now()
start_date = end_date - timedelta(days=30)
timestamps = np.random.randint(int(start_date.timestamp()), int(end_date.timestamp()), size=n_events)
quantities = np.random.randint(1, 5, size=n_events)

events_df = pd.DataFrame(
    {
        "user_id": event_user_ids,
        "item_id": event_item_ids,
        "ts": timestamps,
        "qty": quantities,
    }
)

# Create items data
categories = [f"Category_{i}" for i in range(n_categories)]
brands = [f"Brand_{i}" for i in range(n_brands)]

item_categories = np.random.choice(categories, size=n_items)
item_brands = np.random.choice(brands, size=n_items)
item_prices = np.random.uniform(10, 100, size=n_items).round(2)
item_img_urls = [f"https://example.com/images/{item_id}.jpg" for item_id in item_ids]

items_df = pd.DataFrame(
    {
        "item_id": item_ids,
        "category": item_categories,
        "brand": item_brands,
        "price": item_prices,
        "img_url": item_img_urls,
    }
)

# Save data
events_df.to_parquet("events.parquet", index=False)
items_df.to_csv("items.csv", index=False)

print(f"Created {n_events} events for {n_users} users and {n_items} items")
print("Events saved to events.parquet")
print("Items saved to items.csv")
