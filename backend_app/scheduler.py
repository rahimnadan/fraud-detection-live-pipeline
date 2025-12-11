import schedule
import time
from datetime import date, timedelta
# from training_pipeline import run_training_pipeline
from training_pipeline import run_training_pipeline
def job():
    print("I'm working...")
    today = date.today()
    prev_day = today - timedelta(days=1)
    # prev_day = today + timedelta(days=1)

    print(f"Today Date : {today}")
    print(f"Next Date  : {prev_day}")
    run_training_pipeline(prev_day,today)


# job()
schedule.every().day.at("10:30").do(job)

while True:
    schedule.run_pending()
    time.sleep(1)