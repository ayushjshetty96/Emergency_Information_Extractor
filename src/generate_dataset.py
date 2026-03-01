import random
import pandas as pd

locations = [
"MG Road Bangalore",
"Silk Board Bangalore",
"KR Market Bangalore",
"Indiranagar Metro Station",
"Majestic Bus Stand",
"Electronic City",
"Whitefield Bangalore",
"NH48 Highway",
"Tumkur Road",
"Hebbal Flyover"
]

incidents = [
"fire",
"accident",
"flood",
"earthquake",
"building collapse",
"crime",
"gas leak"
]

severity_levels = ["low","medium","high"]

templates = [

"{incident} near {location}",

"{severity} {incident} near {location}",

"{incident} near {location} {people} injured",

"HELP {incident} happening at {location}",

"hello there is a {incident} near {location}",

"{location} pe {incident} ho gaya",

"{location} alli {incident} aagide"

]

people = [
"one person",
"two people",
"three people",
"several people"
]

data = []

for i in range(5000):

    location = random.choice(locations)
    incident = random.choice(incidents)
    template = random.choice(templates)

    severity = random.choice(severity_levels)
    p = random.choice(people)

    text = template.format(
        location=location,
        incident=incident,
        severity=severity,
        people=p
    )

    data.append({
        "text": text,
        "incident": incident,
        "location": location,
        "severity": severity
    })

df = pd.DataFrame(data)

df.to_csv("emergency_dataset.csv",index=False)

print("Dataset generated!")

def add_noise(text):

    text = text.replace("accident","acident")

    text = text.replace("Bangalore","banglore")

    return text
    