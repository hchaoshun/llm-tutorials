# OpenAI API教程

OpenAI API 是由OpenAI公司开发，为LLM开发人员提供的一个简单接口。通过此API能在应用程序中方便地调用OpenAI提供的大模型基础能力。
OpenAI的API协议已成为LLM领域的标准。

本文将首先介绍OpenAI API基础知识和模型，然后以Chat Completions API和Embedding API为例子介绍OpenAI API的用法。
最后使用Embedding模型构建一个网站智能问答系统。本文内容包括：
- API快速入门
- OpenAI提供的模型
- Chat Completions API和Embedding API
- 基于Embedding模型构建智能问答系统

## API快速入门
- 安装OpenAI Python Library
```shell
pip install --upgrade openai
```
- 设置API Key
```shell
export OPENAI_API_KEY='your-api-key-here'
```

- 发送请求
    - python
      ```python
      from openai import OpenAI
      client = OpenAI()
      # 请求参数将在下文中介绍
      completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
          {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
          {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
        ]
      )
      
      print(completion.choices[0].message)
      ```
  - curl
    ```shell
    curl https://api.openai.com/v1/chat/completions   -H "Content-Type: application/json"   -H "Authorization: Bearer $OPENAI_API_KEY"   -d '{
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "system",
          "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."
        },
        {
          "role": "user",
          "content": "Compose a poem that explains the concept of recursion in programming."
        }
      ]
    }'
    ```
  
## OpenAI提供的模型
OpenAI API 提供不同功能和价位的模型。您甚至可以根据自己的具体使用情况对模型进行定制和微调。

| 模型                         | 描述 |                    常用模型 |
|:---------------------------|:--:|------------------------:|
| GPT-4 and GPT-4 Turbo      | 一套在 GPT-3.5 基础上改进的模型，可理解并生成自然语言或代码 |     gpt-4-turbo-preview |
| GPT-3.5 Turbo              | 一套在 GPT-3.5 基础上改进的模型，可理解并生成自然语言或代码 |          gpt-3.5-turbo	 |
| DALL·E                     | 能根据自然语言提示生成和编辑图像的模型 |               dall-e-3	 |
| TTS                        | 一套可将文本转换成自然发音口语音频的模型 |                  tts-1	 |
| Whisper                    | 可以将音频转换为文本的模型 |               whisper-1 |
| Embeddings                 | 一套可将文本转换为数字形式的模型 | text-embedding-3-large	 |
| Moderation                 | 可检测文本是否敏感或不安全的微调模型 | text-moderation-latest	 |
| GPT base                   | 一套不遵循指令的模型，可理解并生成自然语言或代码 |            babbage-002	 |

## Chat Completions API
Chat Model将对话信息以列表的形式作为输入，并将模型生成的信息作为输出返回。
虽然聊天格式是为了方便多轮对话而设计的，但它同样适用于没有任何对话的单轮任务。
### Example Reqeust
- curl request
```shell
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "model": "gpt-3.5-turbo",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Who won the world series in 2020?"
      },
      {
        "role": "assistant",
        "content": "The Los Angeles Dodgers won the World Series in 2020."
      },
      {
        "role": "user",
        "content": "Where was it played?"
      }
    ]
  }'
```
- python request
```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
    {"role": "user", "content": "Where was it played?"}
  ]
)

print(completion.choices[0].message)
```

Request主要输入是`messages`参数。`messages`必须是一个对象数组，其中每个对象都有一个角色（"system"、"user"或 "assistant"）和`content`。对话可以短至一条信息，也可以来回多次。

通常情况下，对话的格式是`system`在前，`user`和`assistant`交替在后。

`system`有助于设置assistant的行为。例如，您可以修改assistant的个性，或对其在整个对话过程中的行为提供具体指导。不过请注意，系统信息是可选的，如果没有系统信息，模型的行为可能与使用 "You are a helpful assistant."这样的通用信息类似。

`user`提供请求或评论，供assistant回复。assistant信息会存储以前的回复，但也可以由您自己编写，以提供所需的行为示例。

当用户引用之前的信息时，包括对话历史记录就显得非常重要。由于模型无法记忆过去的请求，因此**所有相关信息都必须作为每次请求中对话历史的一部分提供**。如果一个对话无法容纳在模型的标记限制内，就需要以某种方式将其缩短。

### Example Response
```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "The 2020 World Series was played in Texas at Globe Life Field in Arlington.",
        "role": "assistant"
      },
      "logprobs": null
    }
  ],
  "created": 1677664795,
  "id": "chatcmpl-7QyqpwdfhqwajicIEznoc6Q47XAyW",
  "model": "gpt-3.5-turbo-0613",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 17,
    "prompt_tokens": 57,
    "total_tokens": 74
  }
}
```
每次响应，都包括一个`finish_reason`. `finish_reason`的可能值是：
- `stop`：API返回完整信息，或由通过 stop 参数提供的stop序列之一终止的消息
- `length`：不完整模型的输出，由于max_tokens参数或限制令牌
- `function_call`：模型决定调用一个函数
- `content_filter`:由于我们的内容过滤器中的标记而省略了内容
- `null`：API响应仍在进行中的或不完整

根据输入的参数不同，模型的返回可能包括不同的信息。

## Embedding API
LLM的Embedding通常用在RAG中，给模型某个特定领域的知识，以提高生成文本的准确性和信息含量。

要获取Embedding，将文本字符串和Embedding模型名称（如 text-embedding-3-small）一起发送到 embeddings API 端点。响应将包含一个Embedding（浮点数列表），
可以提取它，保存在矢量数据库中，并用于许多不同的用例。
### Example Request
- curl
```shell
curl https://api.openai.com/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "input": "Your text string goes here",
    "model": "text-embedding-3-small"
  }'
```
- python
```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    input="Your text string goes here",
    model="text-embedding-3-small"
)

print(response.data[0].embedding)
```

### Example Response
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [
        -0.006929283495992422,
        -0.005336422007530928,
        ... (omitted for spacing)
        -4.547132266452536e-05,
        -0.024047505110502243
      ],
    }
  ],
  "model": "text-embedding-3-small",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```