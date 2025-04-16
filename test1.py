import os
from openai import OpenAI
import instructor
from pydantic import BaseModel

# Initialize with API key
client = OpenAI(api_key="api_key", base_url="http://localhost:8000/v1")

# Enable instructor patches for OpenAI client
client = instructor.from_openai(client)

class User(BaseModel):
    name: str
    age: int

# Create structured output
user = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "user", "content": "Extract: Jason is 25 years old"},
    ],
)

print(user)
#> User(name='Jason', age=25)