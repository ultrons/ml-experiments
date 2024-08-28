from flask import Flask, render_template, request, redirect
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import datetime
import os
import pickle
import logging
import pytz
import gspread
from oauth2client.service_account import ServiceAccountCredentials

import re

def clean_description(description):
    # Remove any HTML tags
    clean_text = re.sub('<[^<]+?>', '', description)
    return clean_text

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# OAuth 2.0 Setup
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive.file'
]
CREDENTIALS_FILE = 'credentials.json'
TOKEN_FILE = 'token.pickle'
PACIFIC_TZ = pytz.timezone('America/Los_Angeles')

def get_calendar_service():
    credentials = None
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'rb') as token:
            credentials = pickle.load(token)

    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            credentials = flow.run_local_server(port=0)

        with open(TOKEN_FILE, 'wb') as token:
            pickle.dump(credentials, token)

    return build('calendar', 'v3', credentials=credentials)

def get_spreadsheet_service():
    creds = None
    # The file token.pickle stores the user's access and refresh tokens and is
    # created automatically when the authorization flow completes for the first time.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    client = gspread.authorize(creds)
    return client

def get_todo_events(selected_date):
    service = get_calendar_service()
    date = datetime.datetime.strptime(selected_date, '%Y-%m-%d')
    start_time = date.replace(hour=7, minute=0, second=0, microsecond=0, tzinfo=PACIFIC_TZ)
    end_time = date.replace(hour=18, minute=0, second=0, microsecond=0, tzinfo=PACIFIC_TZ)
    start_time_utc = start_time.astimezone(pytz.utc).isoformat()
    end_time_utc = end_time.astimezone(pytz.utc).isoformat()

    events_result = service.events().list(
        calendarId='primary',
        timeMin=start_time_utc,
        timeMax=end_time_utc,
        singleEvents=True,
        orderBy='startTime'
    ).execute()

    events = events_result.get('items', [])
    todo_events = [event for event in events if 'attendees' not in event or not event['attendees']]
    return todo_events

def move_event_to_next_available_slot(event):
    service = get_calendar_service()

    # Extract the original start and end times
    original_start = datetime.datetime.fromisoformat(event['start']['dateTime'].replace('Z', '')).astimezone(PACIFIC_TZ)
    original_end = datetime.datetime.fromisoformat(event['end']['dateTime'].replace('Z', '')).astimezone(PACIFIC_TZ)

    # Calculate the original duration
    original_duration = original_end - original_start

    start_time = datetime.datetime.now(PACIFIC_TZ).replace(hour=7, minute=0)
    end_time = datetime.datetime.now(PACIFIC_TZ).replace(hour=18, minute=0)
    days_ahead = 1

    while True:
        day_start = start_time + datetime.timedelta(days=days_ahead)
        day_end = end_time + datetime.timedelta(days=days_ahead)
        day_start_utc = day_start.astimezone(pytz.utc).isoformat()
        day_end_utc = day_end.astimezone(pytz.utc).isoformat()

        events_result = service.events().list(
            calendarId='primary',
            timeMin=day_start_utc,
            timeMax=day_end_utc,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        busy_times = []
        for e in events_result.get('items', []):
            start = e['start'].get('dateTime') or e['start'].get('date')
            end = e['end'].get('dateTime') or e['end'].get('date')
            if start and end:
                start = datetime.datetime.fromisoformat(start.replace('Z', '')).astimezone(PACIFIC_TZ)
                end = datetime.datetime.fromisoformat(end.replace('Z', '')).astimezone(PACIFIC_TZ)
                busy_times.append((start, end))

        # Check for free slots between 7 AM and 6 PM
        potential_start = day_start
        for start, end in busy_times:
            if potential_start + original_duration <= start:
                # Found a free slot with the original duration
                event['start'] = {'dateTime': potential_start.astimezone(pytz.utc).isoformat(), 'timeZone': 'UTC'}
                event['end'] = {'dateTime': (potential_start + original_duration).astimezone(pytz.utc).isoformat(), 'timeZone': 'UTC'}

                # Update the event with the same duration as the original
                service.events().update(calendarId='primary', eventId=event['id'], body=event).execute()
                logging.info(f"Rescheduled event: {event['summary']} to {potential_start} with original duration ({original_duration})")
                return

            potential_start = end

        logging.info(f"No free slots available on {day_start.date()}. Trying the next day.")
        days_ahead += 1

def move_event_to_next_available_slot_old(event):
    service = get_calendar_service()
    start_time = datetime.datetime.now(PACIFIC_TZ).replace(hour=7, minute=0)
    end_time = datetime.datetime.now(PACIFIC_TZ).replace(hour=18, minute=0)
    days_ahead = 1

    while True:
        day_start = start_time + datetime.timedelta(days=days_ahead)
        day_end = end_time + datetime.timedelta(days=days_ahead)
        day_start_utc = day_start.astimezone(pytz.utc).isoformat()
        day_end_utc = day_end.astimezone(pytz.utc).isoformat()

        events_result = service.events().list(
            calendarId='primary',
            timeMin=day_start_utc,
            timeMax=day_end_utc,
            singleEvents=True,
            orderBy='startTime'
        ).execute()

        busy_times = []
        for e in events_result.get('items', []):
            start = e['start'].get('dateTime') or e['start'].get('date')
            end = e['end'].get('dateTime') or e['end'].get('date')
            if start and end:
                start = datetime.datetime.fromisoformat(start.replace('Z', '')).astimezone(PACIFIC_TZ)
                end = datetime.datetime.fromisoformat(end.replace('Z', '')).astimezone(PACIFIC_TZ)
                busy_times.append((start, end))

        potential_start = day_start
        for start, end in busy_times:
            if potential_start + datetime.timedelta(minutes=30) <= start:
                event['start'] = {'dateTime': potential_start.astimezone(pytz.utc).isoformat(), 'timeZone': 'UTC'}
                event['end'] = {'dateTime': (potential_start + datetime.timedelta(minutes=30)).astimezone(pytz.utc).isoformat(), 'timeZone': 'UTC'}
                event['colorId'] = '6'
                service.events().update(calendarId='primary', eventId=event['id'], body=event).execute()
                return
            potential_start = end
        days_ahead += 1

def export_checked_events_to_sheet(selected_date, todo_events):
    # Get the spreadsheet client
    client = get_spreadsheet_service()

    # Open the existing spreadsheet by name
    #sheet = client.open("Task-Report").sheet1  # Assuming the report is on the first sheet
    sheet = client.open_by_key("1iiqVZNE0nwevkTGT99v26YOX9O1VHk_ZdkTV2oH98GY").sheet1  # Assuming the report is on the first sheet

    for event in todo_events:
        if event['id']  in request.form.getlist('unchecked'):  # Only export checked (completed) tasks
            event_date = selected_date
            attachment = event.get('attachments', [{}])[0].get('fileUrl', 'N/A')
            summary = event['summary']
            description = event.get('description', 'No description provided')
            row = [event_date, attachment, summary, clean_description(description)]
            sheet.append_row(row)

@app.route('/', methods=['GET', 'POST'])
def index():
    selected_date = request.form.get('date') or datetime.datetime.now(PACIFIC_TZ).strftime('%Y-%m-%d')

    if request.method == 'POST':
        if 'fetch' in request.form:
            todo_events = get_todo_events(selected_date)
            return render_template('index.html', events=todo_events, selected_date=selected_date)

        if 'rollover' in request.form:
            todo_events = get_todo_events(selected_date)
            all_event_ids = [event['id'] for event in todo_events]
            checked_ids = request.form.getlist('unchecked')
            unchecked_ids = [event_id for event_id in all_event_ids if event_id not in checked_ids]

            for event in todo_events:
                if event['id'] in unchecked_ids:
                    move_event_to_next_available_slot(event)

            return redirect('/')

        if 'export' in request.form:
            todo_events = get_todo_events(selected_date)
            export_checked_events_to_sheet(selected_date, todo_events)
            return redirect('/')

    todo_events = get_todo_events(selected_date)
    return render_template('index.html', events=todo_events, selected_date=selected_date)

if __name__ == '__main__':
    app.run(debug=True)

