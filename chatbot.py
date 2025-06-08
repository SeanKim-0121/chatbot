import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Levenshtein_Distance import calc_distance # 레벤슈타인 거리 계산 함수 가져오기

class SimpleChatBot:
    def __init__(self, filepath):
        self.questions, self.answers = self.load_data(filepath)
        self.vectorizer = TfidfVectorizer()
        self.question_vectors = self.vectorizer.fit_transform(self.questions)  # 질문을 TF-IDF로 변환

    def load_data(self, filepath):
        data = pd.read_csv(filepath)
        questions = data['Q'].tolist()  # 질문열만 뽑아 파이썬 리스트로 저장
        answers = data['A'].tolist()   # 답변열만 뽑아 파이썬 리스트로 저장
        return questions, answers

    def find_best_answer(self, input_sentence):
        distances = [calc_distance(input_sentence, q) for q in self.questions] # 학습 데이터에 저장된 모든 질문 리스트 (self.questions)에 대해 반복하고 그 결과를 리스트로 만들어서 distances에 저장
        best_match_index = distances.index(min(distances))  # 최소 거리의 인덱스
        
        return self.answers[best_match_index]

# CSV 파일 경로 지정
filepath = os.path.join(os.path.dirname(__file__), 'ChatbotData.csv')

# 간단한 챗봇 인스턴스를 생성
chatbot = SimpleChatBot(filepath)

# '종료'라는 단어가 입력될 때까지 챗봇과의 대화를 반복합니다.
while True:
    input_sentence = input('You: ')
    if input_sentence.lower() == '종료':
        break
    response = chatbot.find_best_answer(input_sentence)
    print('Chatbot:', response)
    