from google import genai
from google.genai import types


def generate():
    client = genai.Client(
        api_key="AIzaSyAy1Qi3tRpV3p2P3xvVDCG6p2IvRd_lUKA",
    )

    model = "gemini-2.0-flash-lite"
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""tổng thống mỹ là ai"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""Tôi được đào tạo đến tháng 6 năm 2024. Do đó, tôi không biết ai là tổng thống Hoa Kỳ sau thời điểm đó. Tuy nhiên, tôi có thể cung cấp cho bạn thông tin liên quan đến trước thời điểm đó.

Tính đến tháng 6 năm 2024, tổng thống Hoa Kỳ là Joe Biden. Xin lưu ý rằng tình hình chính trị có thể thay đổi, và thông tin này có thể không còn chính xác."""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""thế còn vợ ông ấy là ai"""),
            ],
        ),
        types.Content(
            role="model",
            parts=[
                types.Part.from_text(text="""Tính đến tháng 6 năm 2024, vợ của tổng thống Hoa Kỳ Joe Biden là Jill Biden.
"""),
            ],
        ),
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text="""Họ có con chưa ?"""),
            ],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        response_mime_type="text/plain",
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        print(chunk.text, end="")

if __name__ == "__main__":
    generate()
