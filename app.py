import json
import os
import re
import subprocess

import yt_dlp
from flask import Flask, render_template, request
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelWithLMHead
from transformers import BartTokenizer, pipeline, set_seed

app = Flask(__name__)
os.environ["HF_TOKEN"] = "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"


def load_models_and_tokenizers():
    tag_tokenizer = AutoTokenizer.from_pretrained("fabiochiu/t5-base-tag-generation")
    tag_model = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-base-tag-generation")

    QA_tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")
    QA_model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-question-generation-ap")

    return tag_tokenizer, tag_model, QA_tokenizer, QA_model


def my_hook(d):
    if d['status'] == 'finished':
        print("Download completed, converting file...")


def video_download(video_url):
    """
    :type video_url: str, YouTube url
    :rtype: str, path of the downloaded audio file in mp3 format
    """
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': '%(title)s.%(ext)s',  # Set the output template
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'progress_hooks': [my_hook],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        # Generate the MP3 filename based on the original title
        mp3_filename = ydl.prepare_filename(info_dict).replace('.webm', '.mp3').replace('.m4a', '.mp3')

    return mp3_filename


def txt_path_extract(mp3_file_path):

    base_name = os.path.splitext(mp3_file_path)[0]
    output_txt_file_path = base_name + ".txt"

    return output_txt_file_path


def save_string_to_file(txt_path, content):
    # txt_path 경로에 content 내용을 저장하는 함수
    with open(txt_path, 'w', encoding='utf-8') as file:
        file.write(content)


def summary_bert(txt_path):
    set_seed(42)

    txt_file_path = txt_path_extract(txt_path)
    with open(txt_file_path, 'r', encoding='utf-8') as file:
        sample_text = file.read()

    # NLTK를 사용하여 문장 분리
    sentences = sent_tokenize(sample_text)
    # 문장을 하나의 문자열로 병합
    sample_text = ' '.join(sentences)
    # 소문자로 변환
    sample_text = sample_text.lower()
    # 불필요한 특수 문자 제거
    sample_text = re.sub(r'[^\w\s]', '', sample_text)
    # BART 토크나이저 초기화
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")

    # 텍스트 토큰화
    tokenized_text = tokenizer(sample_text, max_length=1024, truncation=True, return_tensors="pt")
    # 디코딩된 텍스트 확인
    decoded_text = tokenizer.decode(tokenized_text['input_ids'][0], skip_special_tokens=True)
    # 요약 파이프라인 초기화 (작은 BART 모델 사용)
    pipe = pipeline("summarization", model="facebook/bart-base")
    # 요약 실행 (배치 크기와 max_length 조절)
    pipe_out = pipe(decoded_text, max_length=600, min_length=300, do_sample=False, batch_size=4)
    # 요약된 텍스트 추출
    summary_text = pipe_out[0]['summary_text']
    # 문장 단위로 분리
    summary_sentences = sent_tokenize(summary_text)

    # 요약된 텍스트 저장
    summaries = {}
    summaries['bart'] = '\n'.join(summary_sentences)
    summaries_txt= summaries['bart']

    return summaries_txt


# 태그 생성 함수
def generate_tags(text, model, tokenizer):
    inputs = tokenizer([text], max_length=512, truncation=True, return_tensors="pt")
    output = model.generate(**inputs, num_beams=8, do_sample=True, min_length=10, max_length=64)
    decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
    tags = list(set(decoded_output.strip().split(", ")))
    return tags


def read_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text


# 질문 생성 함수
def get_questions(answer_list, context, model, tokenizer, max_length=512):
    questions = []
    for answer in answer_list:
        input_text = f"answer: {answer}  context: {context} </s>"
        features = tokenizer([input_text], return_tensors='pt')
        output = model.generate(
            input_ids=features['input_ids'],
            attention_mask=features['attention_mask'],
            max_length=max_length
        )
        question = tokenizer.decode(output[0], skip_special_tokens=True)
        questions.append(question.replace('question: ', ''))
    return questions


# JSON 파일로 저장하는 함수
def save_qa_to_json(questions, tags, filename='result.json'):
    cleaned_questions = [q.replace('question: ', '') for q in questions]
    data = [{"question": q, "answer": a} for q, a in zip(cleaned_questions, tags)]
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


@app.route('/')
def index():
    return render_template('front.html')


@app.route('/download', methods=['POST'])
def download():
    video_url = request.form['video_url']
    mp3_file_path = video_download(video_url)
    print(f"Downloaded MP3 file path: {mp3_file_path}")

    subprocess.run(["whisper", mp3_file_path, "--model", "medium", "--output_format", "txt"])
    txt_file = txt_path_extract(mp3_file_path)

    tag_tokenizer, tag_model, QA_tokenizer, QA_model = load_models_and_tokenizers()
    text = summary_bert(txt_file)
    # 태그 생성
    tags = generate_tags(text, tag_model, tag_tokenizer)
    # 질문 생성
    questions = get_questions(tags, text, QA_model, QA_tokenizer)
    # 결과를 JSON 파일로 저장
    save_qa_to_json(questions, tags)
    # Read the data from the JSON file
    with open('result.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    return render_template('result.html', message='', data=data)

if __name__ == '__main__':
    app.run(debug=True)
