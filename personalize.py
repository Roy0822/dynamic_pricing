import datetime
import random

class UserProfile:
    def __init__(self, user_id):
        self.user_id = user_id
        self.login_times = []
        self.viewed_categories = {}  # Category: count
        self.purchased_categories = {} # Category: count

    def record_login(self):
        now = datetime.datetime.now()
        # Simulate login time in Banqiao, Taiwan (UTC+8)
        banqiao_offset = datetime.timedelta(hours=8)
        banqiao_time = now + banqiao_offset
        self.login_times.append(banqiao_time.strftime("%Y-%m-%d %H:%M:%S"))

    def record_viewed_category(self, category):
        self.viewed_categories[category] = self.viewed_categories.get(category, 0) + 1

    def record_purchase(self, category):
        self.purchased_categories[category] = self.purchased_categories.get(category, 0) + 1

    def get_most_likely_online_hour(self):
        if not self.login_times:
            return None
        hours = [datetime.datetime.strptime(lt, "%Y-%m-%d %H:%M:%S").hour for lt in self.login_times]
        if not hours:
            return None
        # Simple way to find the most frequent hour
        hour_counts = {}
        for hour in hours:
            hour_counts[hour] = hour_counts.get(hour, 0) + 1
        return max(hour_counts, key=hour_counts.get)

    def get_top_interest_categories(self, top_n=2):
        viewed_interests = sorted(self.viewed_categories.items(), key=lambda item: item[1], reverse=True)
        purchased_interests = sorted(self.purchased_categories.items(), key=lambda item: item[1], reverse=True)

        # Combine and weigh viewed and purchased categories (purchases have more weight)
        combined_interests = {}
        for cat, count in viewed_interests:
            combined_interests[cat] = combined_interests.get(cat, 0) + count
        for cat, count in purchased_interests:
            combined_interests[cat] = combined_interests.get(cat, 0) + 2 * count # Give purchase more weight

        top_categories = sorted(combined_interests.items(), key=lambda item: item[1], reverse=True)
        return [cat for cat, count in top_categories[:top_n]]

class NotificationSystem:
    def __init__(self):
        self.users = {}
        self.products_in_category = {
            "Electronics": ["Smartphone X", "Wireless Headphones", "Smartwatch"],
            "Books": ["Mystery Novel", "Science Fiction Epic", "Cookbook"],
            "Apparel": ["T-Shirt", "Jeans", "Sneakers"],
            "Home Goods": ["Coffee Maker", "Throw Pillow", "Desk Lamp"]
        }

    def register_user(self, user_id):
        if user_id not in self.users:
            self.users[user_id] = UserProfile(user_id)
        return self.users[user_id]

    def record_user_activity(self, user_id, activity_type, details=None):
        user = self.users.get(user_id)
        if not user:
            return

        if activity_type == "login":
            user.record_login()
        elif activity_type == "view_category":
            if details and "category" in details:
                user.record_viewed_category(details["category"])
        elif activity_type == "purchase":
            if details and "category" in details:
                user.record_purchase(details["category"])

    def send_personalized_notifications(self, user_id):
        user_profile = self.users.get(user_id)
        if not user_profile:
            return

        likely_online_hour = user_profile.get_most_likely_online_hour()
        top_interests = user_profile.get_top_interest_categories()
        now_banqiao = datetime.datetime.now() + datetime.timedelta(hours=8)

        notifications = []

        # Notification based on online time (simplified: send if current hour is likely online hour)
        if likely_online_hour is not None and now_banqiao.hour == likely_online_hour:
            notifications.append(f"Hi {user_id}! We noticed you're often online around this time. Check out what's new!")

        # Notification based on category interest
        if top_interests:
            for category in top_interests:
                if category in self.products_in_category:
                    sample_product = random.choice(self.products_in_category[category])
                    notifications.append(f"Based on your interest in {category}, you might like our new {sample_product}!")

        if notifications:
            print(f"Sending notifications to user {user_id} at {now_banqiao.strftime('%Y-%m-%d %H:%M:%S')} (Banqiao Time):")
            for notification in notifications:
                print(f"- {notification}")
        else:
            print(f"No relevant notifications to send to user {user_id} right now.")

# Simulate user interactions
notification_system = NotificationSystem()

user1 = notification_system.register_user("Alice")
user2 = notification_system.register_user("Bob")

# Alice's activity
for _ in range(5):
    user1.record_login()
    user1.record_viewed_category("Electronics")
    user1.record_viewed_category("Electronics")
    user1.record_viewed_category("Books")
user1.record_purchase("Electronics")
user1.record_purchase("Electronics")
user1.record_purchase("Books")

# Bob's activity
for _ in range(3):
    user2.record_login()
    user2.record_viewed_category("Apparel")
    user2.record_viewed_category("Apparel")
user2.record_purchase("Apparel")
user2.record_viewed_category("Home Goods")

print("\n--- User Profiles ---")
print(f"Alice's Login Times: {notification_system.users['Alice'].login_times}")
print(f"Alice's Viewed Categories: {notification_system.users['Alice'].viewed_categories}")
print(f"Alice's Purchased Categories: {notification_system.users['Alice'].purchased_categories}")
print(f"Alice's Likely Online Hour: {notification_system.users['Alice'].get_most_likely_online_hour()}")
print(f"Alice's Top Interests: {notification_system.users['Alice'].get_top_interest_categories()}")

print(f"\nBob's Login Times: {notification_system.users['Bob'].login_times}")
print(f"Bob's Viewed Categories: {notification_system.users['Bob'].viewed_categories}")
print(f"Bob's Purchased Categories: {notification_system.users['Bob'].purchased_categories}")
print(f"Bob's Likely Online Hour: {notification_system.users['Bob'].get_most_likely_online_hour()}")
print(f"Bob's Top Interests: {notification_system.users['Bob'].get_top_interest_categories()}")

print("\n--- Sending Notifications ---")
notification_system.send_personalized_notifications("Alice")
notification_system.send_personalized_notifications("Bob")

# Simulate time passing and more logins to potentially trigger online time notification
for _ in range(3):
    now = datetime.datetime.now()
    banqiao_offset = datetime.timedelta(hours=8)
    banqiao_time = now + banqiao_offset
    if banqiao_time.hour == notification_system.users['Alice'].get_most_likely_online_hour():
        print("\n--- Sending Notifications (Simulating Alice being online) ---")
        notification_system.send_personalized_notifications("Alice")
        break
    import time
    time.sleep(3600) # Wait for an hour
