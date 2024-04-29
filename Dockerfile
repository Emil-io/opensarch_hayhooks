FROM deepset/haystack:base-v2.0.1

EXPOSE 1416

RUN pip install opensearch-haystack hayhooks sentence-transformers torch transformers huggingface_hub torchvision torchaudio markdown-it-py mdit_plain pypdf

ENV HF_API_TOKEN=${OPENAI_API_KEY}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

CMD ["hayhooks", "run", "--pipelines-dir", "/pipelines", "--host", "0.0.0.0"]
