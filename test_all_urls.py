import subprocess
import json
import os

# List of 100 URLs (50 fraudulent, 50 legitimate)
urls = [
    "upi://pay?pa=fraud1@unknown&pn=Verify1&am=1000.00&tn=UrgentPayment",
    "upi://pay?pa=fraud2@unknown&pn=SecureLogin2&am=500.00&tn=AccountUpdate",
    # ... (add all 50 fraudulent URLs up to fraud50@unknown)
    "upi://pay?pa=fraud50@unknown&pn=SecureVerify50&am=1000.00&tn=PaymentNow",
    "upi://pay?pa=user1@sbi&pn=User1&am=50.25&tn=Groceries",
    "upi://pay?pa=user2@paytm&pn=User2&am=75.50&tn=BillPayment",
    # ... (add all 50 legitimate URLs up to user50@paytm)
    "upi://pay?pa=user50@paytm&pn=User50&am=35.50&tn=Clothing"
]

results = []
for url in urls:
    print(f"Testing URL: {url}")
    try:
        process = subprocess.run(["python", "test_model.py", url], capture_output=True, text=True, check=True)
        if os.path.exists("results.json"):
            with open("results.json", "r") as f:
                current_results = json.load(f)
                results.extend(current_results)
            os.remove("results.json")  # Clear after reading
        else:
            print(f"No results.json generated for {url}")
    except subprocess.CalledProcessError as e:
        print(f"Error processing {url}: {e.stderr}")

# Save all results
with open("all_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("All results saved to all_results.json")