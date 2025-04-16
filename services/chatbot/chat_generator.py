import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.api_schema import ChatMessage, ChatMessageRole
from typing import Optional, List, AsyncGenerator
from langfuse.decorators import observe
from openai import OpenAI
import openai
from vertexai.generative_models import GenerativeModel, Part, Content, FinishReason
import instructor
import vertexai
import vertexai.preview.generative_models as generative_models
from google import genai
from google.genai import types

# Cấu hình an toàn
safety_settings = {
    generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
}



# Cấu hình tạo nội dung
generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.2,
    "top_p": 0.95,
}


class ChatGenerator:
    async def run(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str],
        temperature: Optional[float],
    ) -> str:
        pass


class OpenAIChatGenerator(ChatGenerator):
    def __init__(
        self,
        model: str = "llama-3.2-1b-instruct",
        api_key: str = "ollama",
        base_url: Optional[str] = "http://localhost:1234/v1",
    ) -> None:
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)



    @observe(name="OpenAIChatGenerator", as_type="generation")
    async def run(self,
                    messages: List[ChatMessage],
                    system_prompt: Optional[str] = None,
                    temperature: Optional[float] = None,
                    response_model: Optional[str] = None,
                    ) -> str:
        try:
            history = []
            if system_prompt is not None:
                history.append({"role": "system", "content": system_prompt})
                
            for message in messages[1:]:
                if message.role == ChatMessageRole.USER:
                    history.append({"role": "user", "content": message.content})
                elif message.role == ChatMessageRole.ASSISTANT:
                    history.append({"role": "assistant", "content": message.content})




            completion = await self.client.beta.chat.completions.parse(
                model=self.model,
                messages=history,
                temperature=temperature if temperature is not None else 0.5,
                response_format=response_model
            )
            response = completion.choices[0].message
            if response.parsed:
                return response.parsed
            elif response.refusal:
                return response.refusal
        
        
        except Exception as e:
            if type(e) == openai.LengthFinishReasonError:
                print("Too many tokens: ", e)
                pass
            else:
                print(e)
                pass
            
            
            
class VertexAIChatGenerator(ChatGenerator):
    def __init__(
        self,
        model: str,
        credentials: str,
        project_id: str = "communi-ai",
        location: str = "asia-southeast1",
    ) -> None:
        vertexai.init(project=project_id, location=location, credentials=credentials)
        self.model_name = model



    @observe(name="VertexAIChatGenerator", as_type="generation")
    async def run(
        self,
        messages: List[ChatMessage],
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        response_model: Optional[str] = None,
    ) -> str:
        
        model = GenerativeModel(
            model_name=self.model_name,
            generation_config=generation_config,
            safety_settings=safety_settings,
            system_instruction=system_prompt,
        )

        client = instructor.from_vertexai(
            client=model,
            mode=instructor.Mode.VERTEXAI_TOOLS,
            _async=True,            
        )

        history: List[Content] = []
        for message in messages[1:]:
            if message.role == ChatMessageRole.USER:
                part = Part.from_text(message.content)  
                history += [Content(parts=[part], role="user")]
            elif message.role == ChatMessageRole.ASSISTANT:
                part = Part.from_text(message.content)
                history += [Content(parts=[part], role="assistant")]

        response = await client.create(
            messages=history,
            response_model=response_model,
            max_retries=5,
            # stream=True,
        )
        
        return response



class GeminiChatGenerator(ChatGenerator):
    def __init__(
        self,
        api_keys: List[str],
        model: str = "gemini-2.0-flash-lite",
    ) -> None:
        self.api_keys = api_keys
        self.model = model

    @observe(name="GeminiChatGenerator", as_type="generation")
    async def run(self,
                messages: List[ChatMessage],
                system_prompt: Optional[str] = None,
                temperature: Optional[float] = None,
                response_model: Optional[str] = None,
                retries: int = 5,
                streaming_partial_json: bool = False,
                ) -> AsyncGenerator[str, None]:
        current_key_index = 0
        gemini_api_key = self.api_keys[current_key_index]
        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="application/json" if response_model else "text/plain",
            response_schema=response_model,
        )
        client = genai.Client(api_key=gemini_api_key)
        client_instructor = instructor.from_genai(
            client,
            mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
        )
        chat_history = [
            types.Content(
                role=message.role,
                parts=[types.Part.from_text(text=message.content)],
            ) for message in messages
        ]

        for attempt in range(retries):
            try:
                if response_model:
                    # existing handling for structured output
                    ...
                else:
                    response = client.models.generate_content_stream(
                        model=self.model,
                        contents=chat_history,
                        config=generate_content_config
                    )

                    for chunk in response:
                        if not chunk.candidates:
                            continue
                        candidate = chunk.candidates[0]

                        if not candidate.content.parts:
                            continue

                        part = candidate.content.parts[0]
                        if part and hasattr(part, "text") and part.text:
                            yield part.text

                        # --- FIXED: break on finish_reason ---
                        if getattr(candidate, "finish_reason", None) == "STOP":
                            return

            except Exception as e:
                print(f"Lỗi khi gọi LLM với API key {gemini_api_key}, thử lại lần {attempt + 1}: {str(e)}")
                if current_key_index >= len(self.api_keys) - 1:
                    print(f"Tất cả các lần thử đều thất bại: {str(e)}")
                    raise e
                current_key_index += 1