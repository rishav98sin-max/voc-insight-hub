# make_data.py
import random
import pandas as pd
from datetime import datetime, timedelta

random.seed(7)

PRODUCT = "ProjectFlow"
personas = ["Founder", "PM", "Ops", "Developer", "Sales"]
plans = ["Free", "Pro", "Team"]
sources = ["Support ticket", "In-app feedback", "NPS comment", "App review", "Sales call note"]

themes = [
    ("Login / SSO issues", [
        "SSO keeps looping me back to the login screen.",
        "Magic link never arrives to my email.",
        "2FA codes are rejected even though they’re correct.",
        "I keep getting logged out every few minutes.",
        "Can’t sign in with Google—button just spins."
    ]),
    ("Invites & permissions", [
        "Inviting teammates fails with 'unknown error'.",
        "I can’t change a user from Viewer to Editor.",
        "Permissions are confusing—who can delete projects?",
        "Need role-based access for clients vs internal staff.",
        "Audit log for permission changes would help."
    ]),
    ("Performance / slowness", [
        "Boards take 10+ seconds to load with large projects.",
        "Search is slow and sometimes times out.",
        "App feels laggy on mobile web.",
        "Switching between projects is noticeably slow.",
        "Gantt view freezes with many tasks."
    ]),
    ("Notifications overload", [
        "Too many email notifications—hard to tune.",
        "I need Slack notifications only for @mentions.",
        "Digest email would be better than individual pings.",
        "Notifications are delayed by hours sometimes.",
        "I want per-project notification settings."
    ]),
    ("Reporting / export", [
        "Need export to CSV for tasks and due dates.",
        "Basic burndown chart would be useful.",
        "Can I schedule a weekly report to email?",
        "Reporting is too limited for stakeholders.",
        "Need time-tracking summary by assignee."
    ]),
    ("Integrations", [
        "Jira integration keeps disconnecting.",
        "Need Google Calendar 2-way sync.",
        "Webhook support for task updates is missing.",
        "Zapier triggers are incomplete.",
        "Integrations page is confusing to set up."
    ]),
    ("Billing & pricing", [
        "Where do I download invoices?",
        "I was charged but the plan didn’t upgrade.",
        "Need prorated billing when adding seats mid-month.",
        "Cancel flow is hard to find.",
        "Can I pay annually with a discount?"
    ]),
    ("Onboarding confusion", [
        "I don’t understand the difference between Workspace and Project.",
        "The app needs a guided setup checklist.",
        "Where do I start after signing up?",
        "Template gallery is hard to discover.",
        "Sample data would help me learn faster."
    ])
]

def jitter(text):
    # small variations to feel “real”
    prefixes = ["", "", "", "Honestly,", "FYI,", "Quick question:", "Not sure if bug but", "Suggestion:"]
    suffixes = ["", "", "", " Please fix asap.", " This is blocking us.", " Any workaround?", " Thanks!"]
    return f"{random.choice(prefixes)} {text}{random.choice(suffixes)}".strip()

def make_rows(n=120):
    rows = []
    start = datetime.now() - timedelta(days=60)

    for i in range(1, n+1):
        theme, samples = random.choice(themes)
        text = jitter(random.choice(samples))

        persona = random.choices(personas, weights=[2,3,2,2,1])[0]
        plan = random.choices(plans, weights=[4,3,2])[0]
        source = random.choice(sources)
        date = (start + timedelta(days=random.randint(0, 60))).date().isoformat()

        # crude “severity”
        severity = random.choices(["Low", "Medium", "High"], weights=[5,3,2])[0]
        if "blocking" in text.lower() or "can't" in text.lower() or "fails" in text.lower():
            severity = random.choices(["Medium", "High"], weights=[2,3])[0]

        rows.append({
            "feedback_id": f"FB-{i:04d}",
            "product": PRODUCT,
            "date": date,
            "source": source,
            "persona": persona,
            "plan": plan,
            "severity": severity,
            "text": text,
            "hidden_theme": theme  # keep this for your own evaluation; drop it in the demo if you want
        })
    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = make_rows(140)
    df.to_csv("sample_feedback.csv", index=False)
    print("Wrote sample_feedback.csv with", len(df), "rows")