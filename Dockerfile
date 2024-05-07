FROM deepset/haystack:base-v2.0.1

EXPOSE 1416

RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/deepset-ai/hayhooks.git /hayhooks \
    && cd /hayhooks \
    && git fetch origin pull/7/head:updated \
    && git checkout updated \
    && pip install .

RUN pip install opensearch-haystack sentence-transformers torch transformers huggingface_hub markdown-it-py mdit_plain pypdf
#RUN pip install 'hayhooks@git+https://github.com/tellmewyatt/hayhooks.git'

ENV HF_API_TOKEN=${HF_API_TOKEN}
ENV OPENAI_API_KEY=${OPENAI_API_KEY}

CMD sleep 5; hayhooks run --pipelines-dir /pipelines --host 0.0.0.0