from typing import Optional, List, AsyncGenerator, Union
from openai import OpenAI
from openai import AsyncOpenAI
from langfuse.decorators import observe
import instructor
from schemas.api_schema import ChatMessage, ChatMessageRole
import vertexai
from vertexai.generative_models import GenerativeModel
import vertexai.generative_models as generative_models
from google import genai
from google.genai import types


# Class Generator cơ bản
class Generator:
    async def run(
        self,
        prompt: str,
        temperature: Optional[float] = None,
    ) -> str:
        pass


class OpenAIGenerator(Generator):
    def __init__(
        self,
        model: str,
        api_key: str = "api_key",
        base_url: Optional[str] = "http://localhost:8000/v1",
    ) -> None:
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key,base_url=base_url)

    
    
    @observe(name="OpenAIGenerator", as_type="generation")
    async def run(self,
                prompt: str,
                temperature: Optional[float] = None,
                response_model: Optional[str] = None,
                retries: int = 5,
                ) -> str:

        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="application/json" if response_model else "text/plain",
            response_schema=response_model,
        )

        client_instructor = instructor.from_openai(
            self.client,
            mode=instructor.Mode.JSON,
        )
        for attempt in range(retries):
            try:
                if response_model:
                    # Non-streaming với response_model - return response
                    response = await client_instructor.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        config=generate_content_config,
                        response_model=response_model,
                    )
                    return response
                else:
                    # Non-streaming không có response_model - return response
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        config=generate_content_config,
                    )
                    return response
            
            except Exception as e:
                print(f"Lỗi khi gọi LLM với model {self.model}, thử lại lần {attempt + 1}: {str(e)}")
                if attempt >= retries - 1:
                    print(f"Tất cả các lần thử đều thất bại: {str(e)}")
                    raise e
                continue


    @observe(name="OpenAIGeneratorStream", as_type="generation")
    async def run_stream(self,
                prompt: str,
                temperature: Optional[float] = None,
                response_model: Optional[str] = None,
                retries: int = 5,
                ) -> AsyncGenerator[str, None]:

        generate_content_config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="application/json" if response_model else "text/plain",
            response_schema=response_model,
        )

        client_instructor = instructor.from_openai(
            self.client,
            mode=instructor.Mode.JSON,
        )
        for attempt in range(retries):
            try:
                if response_model:
                    # Non-streaming với response_model - return response
                    response = await client_instructor.chat.completions.create_partial(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        config=generate_content_config,
                        response_model=response_model,
                    )
                    for chunk in response:
                        yield chunk
                else:
                    # Non-streaming không có response_model - return response
                    response = await self.client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        config=generate_content_config,
                        stream=True,
                    )
                    for chunk in response:
                        yield chunk.choices[0].delta.content
            
            except Exception as e:
                print(f"Lỗi khi gọi LLM với model {self.model}, thử lại lần {attempt + 1}: {str(e)}")
                if attempt >= retries - 1:
                    print(f"Tất cả các lần thử đều thất bại: {str(e)}")
                    raise e
                continue


            

class VertexAIGenerator(Generator):
    def __init__(
        self,
        model: str,
        credentials: str,
        project_id: str = "communi-ai",
        location: str = "asia-southeast1",
    ) -> None:
        
        vertexai.init(project=project_id, location=location, credentials=credentials)
        self.model = GenerativeModel(
            model_name=model,
            generation_config={
                "max_output_tokens": 8192,
                "temperature": 0.5,
                "top_p": 0.95,
            },
            safety_settings={
                generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
                generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            }
        )

    @observe(name="VertexAIGenerator", as_type="generation")
    async def run(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        response_model: Optional[str] = None,
    ) -> str:
        client = instructor.from_vertexai(
            client=self.model,
            mode=instructor.Mode.VERTEXAI_TOOLS,
            _async=True,
        )
        prompt = [
            {
                "role": "user",
                "content": prompt
            }
        ]
        response = await client.create(
            messages=prompt,
            response_model=response_model,
            max_retries=5,
            # stream=True,
        )
        return response

class GeminiGenerator(Generator):
    def __init__(
        self,
        api_keys: List[str],
        model: str = "gemini-2.0-flash",
    ) -> None:
        self.api_keys = api_keys
        self.model = model

    def _convert_role(self, role: ChatMessageRole) -> str:
        # Chuyển đổi role từ format chuẩn sang format của Gemini
        role_mapping = {
            ChatMessageRole.USER: "user",
            ChatMessageRole.ASSISTANT: "model",
            ChatMessageRole.SYSTEM: "system"
        }
        return role_mapping[role]

    @observe(name="GeminiGenerator", as_type="generation")
    async def run(
        self,
        prompt: str,
        messages: Optional[List[ChatMessage]] = None,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = 0.5,
        response_model: Optional[str] = None,
        retries: int = 5,
    ) -> str:
        
        current_key_index = 0
        gemini_api_key = self.api_keys[current_key_index]
        
        if response_model:
            generate_content_config = types.GenerateContentConfig(
                temperature= temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_mime_type="application/json",
                response_schema=response_model,
                system_instruction=system_prompt,
            )
        else:
            generate_content_config = types.GenerateContentConfig(
                temperature=temperature,
                top_p=0.95,
                top_k=40,
                max_output_tokens=8192,
                response_mime_type="text/plain",
                system_instruction=system_prompt,
            )

        client = genai.Client(api_key=gemini_api_key)
        client_instructor = instructor.from_genai(
            client,
            mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
        )
        
        for attempt in range(retries):
            try:
                if messages:
                    if response_model:
                        response = client_instructor.chat.completions.create(
                            model=self.model,
                            messages=[
                                {
                                    "role": message.role,
                                    "content": message.content
                                }
                                for message in messages.append({"role": "user", "content": prompt})
                            ],
                            config=generate_content_config,
                            response_model=response_model,
                        )
                        return response
                    else:
                        response = client.models.generate_content(
                            model=self.model,
                            contents=[
                                types.Content(
                                    role=message.role,
                                    parts=[types.Part.from_text(text=message.content)]
                                )
                                for message in messages.append({"role": "user", "content": prompt})
                            ],
                            config=generate_content_config
                        )
                        return response.text
                else:
                    if response_model:
                        response = client_instructor.chat.completions.create(
                            model=self.model,
                            messages=[{"role": "user", "content": prompt}],
                            config=generate_content_config,
                            response_model=response_model,
                        )
                        return response
                    else:
                        response = client.models.generate_content(
                            model=self.model,
                            contents=[
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_text(text=prompt)]
                                )
                            ],
                            config=generate_content_config
                        )
                        return response.text
            except Exception as e:
                print(f"Lỗi khi gọi LLM với API key {gemini_api_key}, thử lại lần {attempt + 1}: {str(e)}")
                # Nếu đã thử tất cả các API key, ném lỗi
                if current_key_index >= len(self.api_keys) - 1:
                    print(f"Tất cả các lần thử đều thất bại: {str(e)}")
                    raise e
                # Tăng chỉ số API key để thử key tiếp theo
                current_key_index += 1
                
                
                
    @observe(name="GeminiGeneratorStream", as_type="generation")
    async def run_stream(
        self,
        prompt: str,
        temperature: Optional[float] = 0.5,
        response_model: Optional[str] = None,
        retries: int = 5,
    ) -> AsyncGenerator[str, None]:
        current_key_index = 0
        generate_content_config = types.GenerateContentConfig(
            temperature= temperature,
            top_p=0.95,
            top_k=40,
            max_output_tokens=8192,
            response_mime_type="application/json",
            response_schema=response_model,
        )

        gemini_api_key = self.api_keys[current_key_index]
        client = genai.Client(api_key=gemini_api_key)
        client_instructor = instructor.from_genai(
            client,
            mode=instructor.Mode.GENAI_STRUCTURED_OUTPUTS
        )
        
        for attempt in range(retries):
            try:
                if response_model:
                    response = client_instructor.chat.completions.create_partial(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        config=generate_content_config,
                        response_model=response_model,
                    )
                    for chunk in response:
                        yield chunk
                else:
                    response = client.models.generate_content_stream(
                        model=self.model,
                        contents=[
                            types.Content(
                                role="user",
                                parts=[types.Part.from_text(prompt)]
                            )
                        ],
                        config=generate_content_config
                    )
                    for chunk in response:
                        yield chunk.text
            except Exception as e:
                print(f"Lỗi khi gọi LLM với API key {gemini_api_key}, thử lại lần {attempt + 1}: {str(e)}")
                # Nếu đã thử tất cả các API key, ném lỗi
                if current_key_index >= len(self.api_keys) - 1:
                    print(f"Tất cả các lần thử đều thất bại: {str(e)}")
                    raise e
                # Tăng chỉ số API key để thử key tiếp theo
                current_key_index += 1
