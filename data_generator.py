import os
import pandas as pd
import random
from faker import Faker
from datetime import datetime, timedelta
from collections import defaultdict

# ------------------- CONFIG -------------------
NUM_USERS = 5  # Change to 1000+ if needed
DAYS = 90  # 3 months
OUTPUT_DIR = 'synthetic_users'
os.makedirs(OUTPUT_DIR, exist_ok=True)
fake = Faker()
random.seed(42)

# ------------------- MAPPINGS -------------------
merchant_category_map = {
    'Amazon': 'Electronics',
    'Walmart': 'Groceries',
    'Starbucks': 'Dining',
    'Apple': 'Electronics',
    'H&M': 'Clothing',
    'IKEA': 'Home',
    'Zara': 'Clothing',
    'Nike': 'Clothing',
    'Adidas': 'Clothing',
    'Target': 'Groceries',
    'McDonald\'s': 'Dining',
    'CVS': 'Health',
    'Walgreens': 'Health',
    'Delta Airlines': 'Travel',
    'Uber': 'Travel'
}
known_merchants = list(merchant_category_map.keys())
fallback_categories = ['Groceries', 'Dining', 'Travel', 'Electronics', 'Clothing', 'Entertainment', 'Health', 'Home']

category_amount_range = {
    'Groceries': (20, 150),
    'Dining': (10, 60),
    'Clothing': (30, 200),
    'Electronics': (100, 1000),
    'Travel': (50, 500),
    'Health': (10, 100),
    'Entertainment': (10, 100),
    'Home': (30, 400)
}

bank_expense_options = ['ATM Withdrawal', 'Bill Payment', 'Wire Transfer', 'Loan Payment']


def get_amount_for_category(category):
    low, high = category_amount_range.get(category, (20, 100))
    return round(random.uniform(low, high), 2)

# ------------------- GENERATOR -------------------


def generate_user_data(user_id):
    start_date = datetime(2023, 1, 1)
    balance = random.uniform(1000, 10000)
    base_salary = round(random.uniform(3000, 7000), 2)
    salary_day = random.randint(1, 5)
    interest_day = random.randint(20, 28)

    bank_records = []
    card_records = []
    card_monthly_totals = defaultdict(float)

    for day_offset in range(DAYS):
        date = start_date + timedelta(days=day_offset)
        date_str = date.strftime('%Y-%m-%d')
        month_key = date.strftime('%Y-%m')

        # Salary (once a month)
        if date.day == salary_day:
            salary = round(base_salary * random.uniform(0.97, 1.03), 2)
            balance += salary
            bank_records.append({
                'date': date_str,
                'income': salary,
                'expense': 0,
                'balance': round(balance, 2),
                'description': 'Salary'
            })

        # Interest (once a month)
        if date.day == interest_day:
            interest = round(random.uniform(2, 10), 2)
            balance += interest
            bank_records.append({
                'date': date_str,
                'income': interest,
                'expense': 0,
                'balance': round(balance, 2),
                'description': 'Interest Payment'
            })

        # Occasionally add bank expenses
        if random.random() < 0.3:  # ~30% of days
            expense = round(random.uniform(20, 150), 2)
            desc = random.choices(
                population=bank_expense_options,
                weights=[0.4, 0.4, 0.15, 0.05],  # Loan is rare
                k=1
            )[0]
            balance -= expense
            bank_records.append({
                'date': date_str,
                'income': 0,
                'expense': expense,
                'balance': round(balance, 2),
                'description': desc
            })

        # Credit card: 0–1 transaction per day max (~40% chance)
        if random.random() < 0.4:
            use_known = random.random() < 0.7
            if use_known:
                business = random.choice(known_merchants)
                category = merchant_category_map[business]
            else:
                business = fake.company()
                category = random.choice(fallback_categories)

            deal_amount = get_amount_for_category(category)
            is_installment = random.random() < 0.2
            num_parts = random.choice([2, 3]) if is_installment else 1
            installment = round(deal_amount / num_parts, 2)

            for m in range(num_parts):
                future_month = (date + timedelta(days=30 * m)).strftime('%Y-%m')
                card_monthly_totals[future_month] += installment

            # Log initial transaction
            card_records.append({
                'date': date_str,
                'business_name': business,
                'amount_of_deal': deal_amount,
                'amount_paid': installment,
                'category': category
            })

    # Add exact credit card payment to bank (monthly)
    for month_key, total in card_monthly_totals.items():
        total = round(total, 2)
        payment_date = datetime.strptime(month_key + '-28', '%Y-%m-%d')
        balance -= total
        bank_records.append({
            'date': payment_date.strftime('%Y-%m-%d'),
            'income': 0,
            'expense': total,
            'balance': round(balance, 2),
            'description': 'Credit Card Payment'
        })

    # Save to CSV
    bank_df = pd.DataFrame(sorted(bank_records, key=lambda x: x['date']))
    card_df = pd.DataFrame(sorted(card_records, key=lambda x: x['date']))

    bank_df.to_csv(os.path.join(OUTPUT_DIR, f'bank_user_{user_id:04d}.csv'), index=False)
    card_df.to_csv(os.path.join(OUTPUT_DIR, f'card_user_{user_id:04d}.csv'), index=False)

# ------------------- RUN SCRIPT -------------------


for user_id in range(1, NUM_USERS + 1):
    generate_user_data(user_id)

print(f"Done! {NUM_USERS} users generated in '{OUTPUT_DIR}' folder.")
